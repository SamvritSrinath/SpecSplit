"""Orchestrator — manages the async draft→verify ping-pong pipeline.

The ``Orchestrator`` is the user-facing entry point. It sends prompts to the
Draft Worker, forwards the resulting token trees to the Target Worker for
verification, and iterates until the maximum output length or round limit
is reached.

Usage::

    python -m specsplit.workers.orchestrator.client --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import inspect
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Mapping
from typing import Any

import grpc
import grpc.aio

from specsplit.core.config import OrchestratorConfig, load_config_file
from specsplit.core.telemetry import Stopwatch, TelemetryLogger
from specsplit.proto import spec_decoding_pb2, spec_decoding_pb2_grpc
from specsplit.workers.orchestrator.pipeline import (
    PipelineResult,
    run_speculative_loop_async,
)

logger = logging.getLogger(__name__)


def _resolve_tokenizer_model(
    requested_model_name: str | None,
    *,
    draft_model_name: str = "",
    target_model_name: str = "",
) -> str:
    """Choose a tokenizer model, preferring the worker model env vars over bare gpt2."""
    requested = requested_model_name or "gpt2"
    if requested != "gpt2":
        return requested

    if target_model_name:
        logger.info(
            "Tokenizer model not explicitly set; using target worker model=%s",
            target_model_name,
        )
        return target_model_name

    if draft_model_name:
        logger.info(
            "Tokenizer model not explicitly set; using draft worker model=%s",
            draft_model_name,
        )
        return draft_model_name

    target_model = os.environ.get("SPECSPLIT_TARGET_MODEL_NAME", "")
    if target_model:
        logger.info(
            "Tokenizer model not explicitly set; using SPECSPLIT_TARGET_MODEL_NAME=%s",
            target_model,
        )
        return target_model

    draft_model = os.environ.get("SPECSPLIT_DRAFT_MODEL_NAME", "")
    if draft_model:
        logger.info(
            "Tokenizer model not explicitly set; using SPECSPLIT_DRAFT_MODEL_NAME=%s",
            draft_model,
        )
        return draft_model

    return requested


def _ttft_from_telemetry(telemetry: list[dict[str, Any]]) -> float:
    """Compute Time-to-First-Token from pipeline telemetry spans."""
    initial_draft_ms = next(
        (span["wall_time_ms"] for span in telemetry if span["operation"] == "initial_draft"),
        0.0,
    )
    first_round_ms = next(
        (
            span["wall_time_ms"]
            for span in telemetry
            if span["operation"] == "overlapped_round"
            and span.get("metadata", {}).get("round_idx") == 0
        ),
        0.0,
    )
    return initial_draft_ms + first_round_ms


def _capture_relevant_environment(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Capture the runtime environment that affects orchestrator experiments."""
    source = env or os.environ
    prefixes = ("SPECSPLIT_ORCH_", "SPECSPLIT_DRAFT_", "SPECSPLIT_TARGET_")
    return {
        key: source[key]
        for key in sorted(source)
        if key.startswith(prefixes)
    }


def _default_telemetry_output_path(
    *,
    started_at: datetime | None = None,
    root: str | Path = "telemetry",
) -> Path:
    """Return the default timestamped telemetry path for an orchestrator run."""
    ts = (started_at or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S.%f%z")
    return Path(root) / f"orchestrator-run-{ts}.json"


def _build_run_report(
    *,
    prompt: str,
    output_text: str,
    result: PipelineResult,
    config: OrchestratorConfig,
    model_name: str,
    run_id: str,
    started_at_iso: str,
    ended_at_iso: str,
    output_path: str,
    telemetry_spans: list[dict[str, Any]],
    telemetry_events: list[dict[str, Any]],
    worker_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a comprehensive per-run report for offline analysis."""
    generated_tokens = len(result.output_tokens)
    sorted_events = sorted(
        telemetry_events,
        key=lambda event: event.get("timestamp_ns", 0),
    )
    return {
        "run_id": run_id,
        "service": "orchestrator",
        "started_at": started_at_iso,
        "ended_at": ended_at_iso,
        "output_path": output_path,
        "model_name": model_name,
        "prompt": {
            "text": prompt,
            "char_length": len(prompt),
        },
        "output": {
            "text": output_text,
            "token_count": generated_tokens,
            "token_ids": list(result.output_tokens),
        },
        "summary": {
            "generated_tokens": generated_tokens,
            "ttft_ms": round(_ttft_from_telemetry(result.telemetry), 4),
            "tpot_ms": round(result.wall_time_ms / max(generated_tokens, 1), 4),
            "average_acceptance_rate": round(result.acceptance_rate, 4),
            "speculation_hit_rate": round(result.speculation_hit_rate, 4),
            "total_network_idle_ms": round(result.network_idle_ms, 4),
            "total_latency_ms": round(result.wall_time_ms, 4),
            "num_rounds": result.total_rounds,
        },
        "effective_config": config.model_dump(),
        "environment": _capture_relevant_environment(),
        "worker_metadata": worker_metadata,
        "per_round_acceptance": list(result.per_round_acceptance),
        "spans": telemetry_spans,
        "timeline_events": sorted_events,
    }


class Orchestrator:
    """Manages the speculative decoding pipeline between Draft and Target workers.

    The orchestrator runs a loop:
        1. Send prompt context to Draft Worker → receive draft tree.
        2. Forward draft tree to Target Worker → receive accepted tokens.
        3. Append accepted tokens to the output.
        4. If a correction token was sampled, append it and reset draft cache.
        5. Repeat until ``max_output_tokens`` or ``max_rounds`` is reached.

    Args:
        config: Orchestrator configuration (addresses, timeouts, limits).
        model_name: HuggingFace model name for the tokenizer. Defaults to
            ``"gpt2"``.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        model_name: str = "gpt2",
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.model_name = _resolve_tokenizer_model(model_name)
        self._telemetry = TelemetryLogger(service_name="orchestrator")
        self._draft_channel: grpc.Channel | None = None
        self._target_channel: grpc.Channel | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._draft_stub: spec_decoding_pb2_grpc.DraftServiceStub | None = None
        self._target_stub: spec_decoding_pb2_grpc.TargetServiceStub | None = None
        self._tokenizer: Any = None
        self._vocab_bridge: Any | None = None
        self._sync_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._draft_worker_model_name: str = ""
        self._target_worker_model_name: str = ""
        self._draft_worker_vocab_size: int = 0
        self._target_worker_vocab_size: int = 0
        self._workers_probed: bool = False
        self._last_run_report: dict[str, Any] | None = None
        self._last_run_started_at: datetime | None = None

        logger.info(
            "Orchestrator initialized (draft=%s, target=%s, tokenizer=%s, max_draft=%d, draft_temp=%.2f)",
            self.config.draft_address,
            self.config.target_address,
            self.model_name,
            self.config.max_draft_tokens,
            self.config.draft_temperature,
        )

    def _ensure_tokenizer(self) -> Any:
        """Lazily load the HuggingFace tokenizer on first use.

        Uses worker-reported Ping metadata for vocabulary checks so the
        orchestrator does not need direct filesystem access to worker model paths.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            worker_vocabs = {
                vocab_size
                for vocab_size in (self._draft_worker_vocab_size, self._target_worker_vocab_size)
                if vocab_size > 0
            }
            if (
                self.model_name == "gpt2"
                and worker_vocabs
                and any(vocab_size != 50257 for vocab_size in worker_vocabs)
            ):
                raise RuntimeError(
                    "Tokenizer model is unresolved: orchestrator is still on gpt2, but workers report "
                    f"non-gpt2 vocab sizes {sorted(worker_vocabs)}. Restart the workers after updating "
                    "their code so Ping returns model metadata, or pass --model-name / "
                    "SPECSPLIT_ORCH_TOKENIZER_MODEL explicitly."
                )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info("Tokenizer loaded: %s", self.model_name)

            # Prefer worker-reported vocab sizes from Ping over loading the
            # draft/target tokenizer paths locally. Those paths may not exist
            # on the orchestrator host even when the workers themselves are healthy.
            draft_model_env = self._draft_worker_model_name or os.environ.get("SPECSPLIT_DRAFT_MODEL_NAME", "")
            target_model_env = self._target_worker_model_name or os.environ.get("SPECSPLIT_TARGET_MODEL_NAME", "")

            if draft_model_env and target_model_env and draft_model_env != target_model_env:
                logger.info(
                    "Draft and Target use different models. Comparing reported vocab sizes from Ping."
                )
                if self._draft_worker_vocab_size > 0 and self._target_worker_vocab_size > 0:
                    if self._draft_worker_vocab_size != self._target_worker_vocab_size:
                        logger.warning(
                            "Draft and Target report different vocab sizes (%d vs %d). "
                            "Proceeding without local tokenizer-path validation; "
                            "token ID mapping is not initialized.",
                            self._draft_worker_vocab_size,
                            self._target_worker_vocab_size,
                        )
                    else:
                        logger.info(
                            "Draft and Target report matching vocab sizes (%d).",
                            self._draft_worker_vocab_size,
                        )
                else:
                    logger.warning(
                        "Draft and Target use different models, but Ping did not provide both vocab sizes. "
                        "Skipping local tokenizer-path validation."
                    )

        return self._tokenizer

    async def _refresh_worker_metadata(self) -> None:
        """Probe the workers for model metadata and adopt it when tokenizer is unset."""
        if self._workers_probed:
            return
        if self._draft_stub is None or self._target_stub is None:
            return

        draft_request_id = f"ping-draft-{uuid.uuid4().hex[:8]}"
        self._telemetry.record_event(
            "rpc_request_sent",
            rpc="Ping",
            peer="draft",
            request_id=draft_request_id,
            timeout_s=self.config.timeout_s,
        )
        draft_sw = Stopwatch().start()
        try:
            draft_ping = self._draft_stub.Ping(spec_decoding_pb2.PingRequest(), timeout=self.config.timeout_s)
            if inspect.isawaitable(draft_ping):
                draft_ping = await draft_ping
        except Exception as exc:
            draft_sw.stop()
            self._telemetry.record_event(
                "rpc_error",
                rpc="Ping",
                peer="draft",
                request_id=draft_request_id,
                elapsed_ms=draft_sw.elapsed_ms,
                error=type(exc).__name__,
                error_details=str(exc),
            )
            logger.warning("Draft worker metadata probe failed: %s", exc)
            draft_ping = None
        else:
            draft_sw.stop()
            self._telemetry.record_event(
                "rpc_response_received",
                rpc="Ping",
                peer="draft",
                request_id=draft_request_id,
                elapsed_ms=draft_sw.elapsed_ms,
                worker_status=getattr(draft_ping, "status", ""),
                model_name=getattr(draft_ping, "model_name", ""),
                vocab_size=int(getattr(draft_ping, "vocab_size", 0) or 0),
            )

        target_request_id = f"ping-target-{uuid.uuid4().hex[:8]}"
        self._telemetry.record_event(
            "rpc_request_sent",
            rpc="Ping",
            peer="target",
            request_id=target_request_id,
            timeout_s=self.config.timeout_s,
        )
        target_sw = Stopwatch().start()
        try:
            target_ping = self._target_stub.Ping(spec_decoding_pb2.PingRequest(), timeout=self.config.timeout_s)
            if inspect.isawaitable(target_ping):
                target_ping = await target_ping
        except Exception as exc:
            target_sw.stop()
            self._telemetry.record_event(
                "rpc_error",
                rpc="Ping",
                peer="target",
                request_id=target_request_id,
                elapsed_ms=target_sw.elapsed_ms,
                error=type(exc).__name__,
                error_details=str(exc),
            )
            logger.warning("Target worker metadata probe failed: %s", exc)
            target_ping = None
        else:
            target_sw.stop()
            self._telemetry.record_event(
                "rpc_response_received",
                rpc="Ping",
                peer="target",
                request_id=target_request_id,
                elapsed_ms=target_sw.elapsed_ms,
                worker_status=getattr(target_ping, "status", ""),
                model_name=getattr(target_ping, "model_name", ""),
                vocab_size=int(getattr(target_ping, "vocab_size", 0) or 0),
            )

        if draft_ping is not None:
            self._draft_worker_model_name = getattr(draft_ping, "model_name", "") or ""
            self._draft_worker_vocab_size = int(getattr(draft_ping, "vocab_size", 0) or 0)
        if target_ping is not None:
            self._target_worker_model_name = getattr(target_ping, "model_name", "") or ""
            self._target_worker_vocab_size = int(getattr(target_ping, "vocab_size", 0) or 0)

        resolved_model_name = _resolve_tokenizer_model(
            self.model_name,
            draft_model_name=self._draft_worker_model_name,
            target_model_name=self._target_worker_model_name,
        )
        if resolved_model_name != self.model_name:
            logger.info(
                "Tokenizer model updated from %s to %s after worker probe",
                self.model_name,
                resolved_model_name,
            )
            self.model_name = resolved_model_name

        self._workers_probed = True

    def connect(self) -> None:
        """Establish async gRPC channels to Draft and Target workers."""
        self._draft_channel = grpc.aio.insecure_channel(self.config.draft_address)
        self._draft_stub = spec_decoding_pb2_grpc.DraftServiceStub(
            self._draft_channel,
        )

        self._target_channel = grpc.aio.insecure_channel(self.config.target_address)
        self._target_stub = spec_decoding_pb2_grpc.TargetServiceStub(
            self._target_channel,
        )

        logger.info("Async gRPC channels established")

    async def close(self) -> None:
        """Close gRPC channels and release resources."""
        if self._draft_channel is not None:
            await self._draft_channel.close()
            self._draft_channel = None
        if self._target_channel is not None:
            await self._target_channel.close()
            self._target_channel = None

        logger.info("Async gRPC channels closed")

    async def run_with_result(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> tuple[str, PipelineResult]:
        """Run the full speculative decoding pipeline for a given prompt.

        Tokenizes the prompt, executes the async speculative loop over
        gRPC, and decodes the resulting tokens back to a string.

        Issue 7: Generates a unique session ID per call when KV caching
        is enabled, preventing cross-prompt cache pollution. Sends
        ``EndSession`` RPC in a ``finally`` block to prevent leaks.

        Args:
            prompt: The user's input text prompt.
            session_id: Optional caller-supplied session ID. If not
                provided and KV caching is enabled, a unique ID is
                generated automatically.

        Returns:
            A tuple of (generated output text, PipelineResult with full metrics).
        """
        logger.info("Starting generation for prompt: %r", prompt[:80])
        self._telemetry.reset()
        started_at = datetime.now().astimezone()
        self._last_run_started_at = started_at
        started_at_iso = started_at.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        run_id = uuid.uuid4().hex[:12]
        self._telemetry.record_event(
            "run_started",
            run_id=run_id,
            prompt_char_length=len(prompt),
            draft_address=self.config.draft_address,
            target_address=self.config.target_address,
        )

        # Create gRPC channels in the current event loop (grpc.aio channels are
        # loop-bound; creating them from sync code causes "attached to different loop")
        if self._draft_channel is None:
            self.connect()

        await self._refresh_worker_metadata()
        tokenizer = self._ensure_tokenizer()
        prompt_ids: list[int] = tokenizer.encode(prompt)
        eos_token_id: int = tokenizer.eos_token_id or 2

        # Issue 7: Generate unique session ID per request (not "default")
        if session_id is None:
            session_id = uuid.uuid4().hex if self.config.use_target_kv_cache else None
        self._telemetry.record_event(
            "prompt_tokenized",
            run_id=run_id,
            session_id=session_id or "",
            prompt_token_count=len(prompt_ids),
            eos_token_id=eos_token_id,
        )

        result: PipelineResult | None = None
        output_text = ""

        try:
            with self._telemetry.span(
                "full_pipeline",
                prompt_len=len(prompt),
                prompt_token_count=len(prompt_ids),
                run_id=run_id,
                session_id=session_id or "",
            ):
                result = await run_speculative_loop_async(
                    draft_stub=self._draft_stub,
                    target_stub=self._target_stub,
                    prompt_ids=prompt_ids,
                    config=self.config,
                    session_id=session_id,
                    eos_token_id=eos_token_id,
                    vocab_bridge=self._vocab_bridge,
                    telemetry=self._telemetry,
                )

                output_text = tokenizer.decode(
                    result.output_tokens,
                    skip_special_tokens=True,
                )

                logger.info(
                    "Pipeline complete: %d tokens in %d rounds, acceptance=%.1f%%, wall_time=%.1f ms",
                    len(result.output_tokens),
                    result.total_rounds,
                    result.acceptance_rate * 100,
                    result.wall_time_ms,
                )
                self._telemetry.record_event(
                    "run_completed",
                    run_id=run_id,
                    session_id=session_id or "",
                    output_token_count=len(result.output_tokens),
                    total_rounds=result.total_rounds,
                    acceptance_rate=result.acceptance_rate,
                    speculation_hit_rate=result.speculation_hit_rate,
                    wall_time_ms=result.wall_time_ms,
                    network_idle_ms=result.network_idle_ms,
                )
        finally:
            # Issue 7: Always clean up the session to prevent KV cache leaks
            if session_id is not None and self._target_stub is not None:
                try:
                    from specsplit.proto import spec_decoding_pb2

                    request_id = f"end-session-{session_id}"
                    end_req = spec_decoding_pb2.EndSessionRequest(
                        session_id=session_id,
                    )
                    self._telemetry.record_event(
                        "rpc_request_sent",
                        rpc="EndSession",
                        peer="target",
                        request_id=request_id,
                        session_id=session_id,
                    )
                    end_resp = self._target_stub.EndSession(end_req)
                    if inspect.isawaitable(end_resp):
                        await end_resp
                    self._telemetry.record_event(
                        "rpc_response_received",
                        rpc="EndSession",
                        peer="target",
                        request_id=request_id,
                        session_id=session_id,
                        was_active=bool(getattr(end_resp, "was_active", False)),
                    )
                    logger.debug("Session ended: %s", session_id)
                except Exception as exc:
                    # EndSession is best-effort cleanup
                    self._telemetry.record_event(
                        "rpc_error",
                        rpc="EndSession",
                        peer="target",
                        request_id=f"end-session-{session_id}",
                        session_id=session_id,
                        error=type(exc).__name__,
                        error_details=str(exc),
                    )
                    logger.debug(
                        "EndSession cleanup failed (non-critical): session=%s",
                        session_id,
                    )

        if result is not None:
            ended_at = datetime.now().astimezone()
            ended_at_iso = ended_at.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            worker_metadata = {
                "draft": {
                    "address": self.config.draft_address,
                    "model_name": self._draft_worker_model_name,
                    "vocab_size": self._draft_worker_vocab_size,
                },
                "target": {
                    "address": self.config.target_address,
                    "model_name": self._target_worker_model_name,
                    "vocab_size": self._target_worker_vocab_size,
                },
            }
            self._last_run_report = _build_run_report(
                prompt=prompt,
                output_text=output_text,
                result=result,
                config=self.config,
                model_name=self.model_name,
                run_id=run_id,
                started_at_iso=started_at_iso,
                ended_at_iso=ended_at_iso,
                output_path="",
                telemetry_spans=[span.to_dict() for span in self._telemetry.spans],
                telemetry_events=[event.to_dict() for event in self._telemetry.events],
                worker_metadata=worker_metadata,
            )

        return output_text, result

    def run_with_result_sync(self, prompt: str) -> tuple[str, PipelineResult]:
        """Synchronous wrapper around :meth:`run_with_result`.

        Creates a new event loop and runs the async method to completion.
        Use this from non-async callers (CLI, benchmarks, etc.).

        When called from an async context (FastAPI, Jupyter), runs the pipeline
        in a dedicated thread with its own loop to avoid "event loop already
        running" deadlock.

        Args:
            prompt: The user's input text prompt.

        Returns:
            A tuple of (generated output text, PipelineResult with full metrics).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to use run_until_complete
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop.run_until_complete(self.run_with_result(prompt))

        # Already inside an async context — run in a dedicated thread to avoid
        # "This event loop is already running" from run_until_complete.
        def _run_in_thread() -> tuple[str, PipelineResult]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.run_with_result(prompt))
            finally:
                loop.close()

        if self._sync_executor is None:
            self._sync_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="orchestrator-sync"
            )
        return self._sync_executor.submit(_run_in_thread).result()

    def run(self, prompt: str) -> str:
        """Run the pipeline and return the generated text.

        Thin wrapper around :meth:`run_with_result_sync` for callers that only
        need the output string.

        Args:
            prompt: The user's input text prompt.

        Returns:
            The generated output text.
        """
        output_text, _ = self.run_with_result_sync(prompt)
        return output_text

    def chat_session(self) -> "ConversationSession":
        """Create a new stateful ConversationSession."""
        if not self.config.use_target_kv_cache:
            logger.warning("Starting chat session with use_target_kv_cache=False. Performance will suffer.")
        self._ensure_tokenizer()
        return ConversationSession(self)

    def export_telemetry(self, path: str | None = None) -> str:
        """Export the most recent run report to a JSON file."""
        if self._last_run_report is None:
            raise RuntimeError("No orchestrator run has completed yet.")

        if path is None:
            out_path = _default_telemetry_output_path(started_at=self._last_run_started_at)
        else:
            candidate = Path(path)
            if candidate.suffix.lower() == ".json":
                out_path = candidate
            else:
                out_path = candidate / _default_telemetry_output_path(
                    started_at=self._last_run_started_at,
                    root=".",
                ).name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = dict(self._last_run_report)
        report["output_path"] = str(out_path)
        out_path.write_text(json.dumps(report, indent=2))
        return str(out_path)


class ConversationSession:
    """A stateful conversation session for multi-turn interactions.

    Maintains accumulated token IDs across multiple `generate()` turns
    to avoid O(n^2) re-tokenization. Automatically manages the session ID
    and KV cache cleanup on the Target Worker via context manager.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        self.session_id: str = uuid.uuid4().hex
        self.accumulated_token_ids: list[int] = []
        self._is_active: bool = True
        logger.info("Initializing ConversationSession %s", self.session_id)

    def __enter__(self) -> "ConversationSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end()

    async def generate_async(self, user_prompt: str) -> str:
        """Async generation for the next turn in the conversation."""
        if not self._is_active:
            raise RuntimeError("Cannot generate on an ended session.")

        tokenizer = self.orchestrator._tokenizer
        eos_token_id = tokenizer.eos_token_id or 2

        # Tokenize new prompt and append to history
        new_prompt_ids = tokenizer.encode(user_prompt)
        self.accumulated_token_ids.extend(new_prompt_ids)

        if self.orchestrator._draft_channel is None:
            self.orchestrator.connect()

        with self.orchestrator._telemetry.span("chat_turn", session_id=self.session_id):
            result: PipelineResult = await run_speculative_loop_async(
                draft_stub=self.orchestrator._draft_stub,
                target_stub=self.orchestrator._target_stub,
                prompt_ids=self.accumulated_token_ids,
                config=self.orchestrator.config,
                session_id=self.session_id,
                eos_token_id=eos_token_id,
                vocab_bridge=self.orchestrator._vocab_bridge,
            )

            self.accumulated_token_ids.extend(result.output_tokens)
            output_text = tokenizer.decode(result.output_tokens, skip_special_tokens=True)
            return output_text

    def generate(self, user_prompt: str) -> str:
        """Sync generation for the next turn in the conversation."""
        orch = self.orchestrator
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if orch._loop is None or orch._loop.is_closed():
                orch._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(orch._loop)
            return orch._loop.run_until_complete(self.generate_async(user_prompt))

        def _run_in_thread() -> str:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.generate_async(user_prompt))
            finally:
                loop.close()

        if orch._sync_executor is None:
            orch._sync_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="orchestrator-sync"
            )
        return orch._sync_executor.submit(_run_in_thread).result()

    def end(self) -> None:
        """End the session and explicitly flush the Target Worker's KV cache."""
        if not self._is_active:
            return
        self._is_active = False

        if self.orchestrator._target_stub is not None:
            cleanup_log_msg = "ConversationSession %s cleanup failed (non-critical)."
            try:
                from specsplit.proto import spec_decoding_pb2

                end_req = spec_decoding_pb2.EndSessionRequest(session_id=self.session_id)
                call = self.orchestrator._target_stub.EndSession(end_req)

                async def _run_end_session() -> None:
                    await call

                def _log_end_session_error(task: asyncio.Task[None]) -> None:
                    try:
                        task.result()
                    except Exception as exc:
                        logger.debug(cleanup_log_msg + " (%s)", self.session_id, exc)

                try:
                    loop = asyncio.get_running_loop()
                    _end_task = loop.create_task(_run_end_session())
                    _end_task.add_done_callback(_log_end_session_error)
                except RuntimeError:
                    asyncio.run(_run_end_session())
                logger.debug("ConversationSession %s ended.", self.session_id)
            except Exception:
                logger.debug(cleanup_log_msg, self.session_id)


def main() -> None:
    """CLI entry point for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="SpecSplit Orchestrator — run speculative decoding pipeline",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum draft→verify rounds (overrides config)",
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        default=None,
        help="Draft tree depth K (overrides config). Lower = higher acceptance, fewer tokens/round.",
    )
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=None,
        help="Draft sampling temperature. 0 = greedy (align with target). Overrides config.",
    )
    parser.add_argument(
        "--verify-temperature",
        type=float,
        default=None,
        help="Verification temperature. Defaults to draft temperature unless explicitly set.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Tokenizer model; must match target/draft (e.g. Qwen2/Qwen2.5-7B-Instruct). Overrides SPECSPLIT_ORCH_TOKENIZER_MODEL.",
    )
    parser.add_argument(
        "--telemetry-output",
        type=str,
        default=None,
        help="Telemetry JSON path or directory. If omitted, a timestamped file is written under telemetry/.",
    )
    parser.add_argument(
        "--use-target-cache",
        action="store_true",
        help="Enable target KV cache (dynamic caching).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output tokens to generate (overrides config, default 1024).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML or JSON config file. CLI args and env vars override file values.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )

    # Load config file if provided (lowest priority after env vars)
    file_cfg: dict = {}
    if args.config:
        all_sections = load_config_file(args.config)
        file_cfg = all_sections.get("orchestrator", {})

    # CLI args override file config; env vars override both (pydantic-settings)
    config_kw: dict = {**file_cfg}
    if args.use_target_cache:
        config_kw["use_target_kv_cache"] = True
    if args.max_rounds is not None:
        config_kw["max_rounds"] = args.max_rounds
    if args.max_output_tokens is not None:
        config_kw["max_output_tokens"] = args.max_output_tokens
    if args.max_draft_tokens is not None:
        config_kw["max_draft_tokens"] = args.max_draft_tokens
    if args.draft_temperature is not None:
        config_kw["draft_temperature"] = args.draft_temperature
    if args.verify_temperature is not None:
        config_kw["verify_temperature"] = args.verify_temperature

    # Temperature lockstep (CLI ergonomics):
    # If verify temperature is not explicitly configured, align it to draft temperature
    # to avoid accidental low-acceptance mixed mode (e.g., verify=0.0, draft=1.0).
    verify_explicit = (
        args.verify_temperature is not None
        or "SPECSPLIT_ORCH_VERIFY_TEMPERATURE" in os.environ
        or "verify_temperature" in file_cfg
    )
    if not verify_explicit and "verify_temperature" not in config_kw:
        draft_temp_for_lockstep: float | None = None
        if args.draft_temperature is not None:
            draft_temp_for_lockstep = args.draft_temperature
        elif "SPECSPLIT_ORCH_DRAFT_TEMPERATURE" in os.environ:
            try:
                draft_temp_for_lockstep = float(
                    os.environ["SPECSPLIT_ORCH_DRAFT_TEMPERATURE"]
                )
            except ValueError:
                draft_temp_for_lockstep = None
        elif "draft_temperature" in file_cfg:
            try:
                draft_temp_for_lockstep = float(file_cfg["draft_temperature"])
            except (TypeError, ValueError):
                draft_temp_for_lockstep = None

        if draft_temp_for_lockstep is not None:
            config_kw["verify_temperature"] = draft_temp_for_lockstep
            logger.info(
                "Verify temperature not explicitly set. Locking verify_temperature=%.2f to draft_temperature.",
                draft_temp_for_lockstep,
            )
    config = OrchestratorConfig(**config_kw)

    # Model name priority: --model-name > config file > OrchestratorConfig.tokenizer_model
    if args.model_name is not None:
        model_name = args.model_name
    elif file_cfg.get("tokenizer_model"):
        model_name = file_cfg["tokenizer_model"]
    else:
        model_name = config.tokenizer_model
    model_name = _resolve_tokenizer_model(model_name)

    # Warn if orchestrator tokenizer differs from worker model env vars
    draft_model_env = os.environ.get("SPECSPLIT_DRAFT_MODEL_NAME", "")
    target_model_env = os.environ.get("SPECSPLIT_TARGET_MODEL_NAME", "")
    worker_model_names = {name for name in (draft_model_env, target_model_env) if name}
    if draft_model_env and target_model_env and model_name not in worker_model_names:
        logger.warning(
            "Tokenizer mismatch: orchestrator uses '%s' but workers use '%s' and '%s'. "
            "Speculative decoding requires identical vocabularies.",
            model_name,
            draft_model_env,
            target_model_env,
        )
    elif draft_model_env and draft_model_env != model_name and not target_model_env:
        logger.warning(
            "Tokenizer mismatch: orchestrator uses '%s' but SPECSPLIT_DRAFT_MODEL_NAME='%s'. "
            "Speculative decoding requires identical vocabularies.",
            model_name,
            draft_model_env,
        )
    elif target_model_env and target_model_env != model_name and not draft_model_env:
        logger.warning(
            "Tokenizer mismatch: orchestrator uses '%s' but SPECSPLIT_TARGET_MODEL_NAME='%s'. "
            "Speculative decoding requires identical vocabularies.",
            model_name,
            target_model_env,
        )

    orch = Orchestrator(config=config, model_name=model_name)

    async def _run() -> tuple[str, PipelineResult]:
        orch.connect()  # Must be inside event loop (grpc.aio channels are loop-bound)
        return await orch.run_with_result(args.prompt)

    output_text, result = asyncio.run(_run())

    print(f"\n{'=' * 60}")
    print("Generated Output:")
    print(f"{'=' * 60}")
    print(output_text)
    print(f"\n{'=' * 60}")
    print("Pipeline Metrics:")
    print(f"{'=' * 60}")
    print(f"  Tokens generated:     {len(result.output_tokens)}")
    print(f"  Rounds:               {result.total_rounds}")
    print(f"  Acceptance rate:      {result.acceptance_rate * 100:.1f}%")
    print(f"  Speculation hit rate: {result.speculation_hit_rate * 100:.1f}%")
    print(f"  Wall time:            {result.wall_time_ms:.1f} ms")
    telemetry_path = orch.export_telemetry(args.telemetry_output)
    print(f"\nTelemetry exported to: {telemetry_path}")


if __name__ == "__main__":
    main()
