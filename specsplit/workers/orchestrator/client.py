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
import logging
import os
import uuid
from typing import Any

import grpc
import grpc.aio

from specsplit.core.config import OrchestratorConfig, load_config_file
from specsplit.core.telemetry import TelemetryLogger
from specsplit.proto import spec_decoding_pb2_grpc
from specsplit.workers.orchestrator.pipeline import (
    PipelineResult,
    run_speculative_loop_async,
)

logger = logging.getLogger(__name__)


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
        self.model_name = model_name
        self._telemetry = TelemetryLogger(service_name="orchestrator")
        self._draft_channel: grpc.Channel | None = None
        self._target_channel: grpc.Channel | None = None
        self._draft_stub: spec_decoding_pb2_grpc.DraftServiceStub | None = None
        self._target_stub: spec_decoding_pb2_grpc.TargetServiceStub | None = None
        self._tokenizer: Any = None

        logger.info(
            "Orchestrator initialized (draft=%s, target=%s, tokenizer=%s)",
            self.config.draft_address,
            self.config.target_address,
            self.model_name,
        )

    def _ensure_tokenizer(self) -> Any:
        """Lazily load the HuggingFace tokenizer on first use."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info("Tokenizer loaded: %s", self.model_name)
        return self._tokenizer

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

        tokenizer = self._ensure_tokenizer()
        prompt_ids: list[int] = tokenizer.encode(prompt)
        eos_token_id: int = tokenizer.eos_token_id or 2

        # Issue 7: Generate unique session ID per request (not "default")
        if session_id is None:
            session_id = uuid.uuid4().hex if self.config.use_target_kv_cache else None

        try:
            with self._telemetry.span("full_pipeline", prompt_len=len(prompt)):
                result: PipelineResult = await run_speculative_loop_async(
                    draft_stub=self._draft_stub,
                    target_stub=self._target_stub,
                    prompt_ids=prompt_ids,
                    config=self.config,
                    session_id=session_id,
                    eos_token_id=eos_token_id,
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
        finally:
            # Issue 7: Always clean up the session to prevent KV cache leaks
            if session_id is not None and self._target_stub is not None:
                try:
                    from specsplit.proto import spec_decoding_pb2

                    end_req = spec_decoding_pb2.EndSessionRequest(
                        session_id=session_id,
                    )
                    end_resp = self._target_stub.EndSession(end_req)
                    if asyncio.iscoroutine(end_resp) or asyncio.isfuture(end_resp):
                        await end_resp
                    logger.debug("Session ended: %s", session_id)
                except Exception:
                    # EndSession is best-effort cleanup
                    logger.debug(
                        "EndSession cleanup failed (non-critical): session=%s",
                        session_id,
                    )

        return output_text, result

    def run_with_result_sync(self, prompt: str) -> tuple[str, PipelineResult]:
        """Synchronous wrapper around :meth:`run_with_result`.

        Creates a new event loop and runs the async method to completion.
        Use this from non-async callers (CLI, benchmarks, etc.).

        Use this from non-async callers (CLI, benchmarks, etc.).
        Note that establishing a single event loop per orchestrator prevents
        leaking and crashing gRPC components tied to differing loop lifetimes.

        Args:
            prompt: The user's input text prompt.

        Returns:
            A tuple of (generated output text, PipelineResult with full metrics).
        """
        # Because we bind to a specific loop in connect() depending on when it's called,
        # calling asyncio.run() repeatedly creates new loops which crashes aio channels.
        # So we should get the current loop or make one and run until complete.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.run_with_result(prompt))

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

    def export_telemetry(self, path: str) -> None:
        """Export collected telemetry spans to a JSON file."""
        self._telemetry.export(path)


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
        "--model-name",
        type=str,
        default=None,
        help="Tokenizer model; must match target/draft (e.g. Qwen2/Qwen2.5-7B-Instruct). Overrides SPECSPLIT_ORCH_TOKENIZER_MODEL.",
    )
    parser.add_argument(
        "--telemetry-output",
        type=str,
        default=None,
        help="Path to export telemetry JSON",
    )
    parser.add_argument(
        "--use-target-cache",
        action="store_true",
        help="Enable target KV cache (dynamic caching). Default is naive/stateless per round for testing.",
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
    config = OrchestratorConfig(**config_kw)

    # Model name priority: --model-name > config file > OrchestratorConfig.tokenizer_model
    if args.model_name is not None:
        model_name = args.model_name
    elif file_cfg.get("tokenizer_model"):
        model_name = file_cfg["tokenizer_model"]
    else:
        model_name = config.tokenizer_model

    # Warn if orchestrator tokenizer differs from worker model env vars
    draft_model_env = os.environ.get("SPECSPLIT_DRAFT_MODEL_NAME", "")
    target_model_env = os.environ.get("SPECSPLIT_TARGET_MODEL_NAME", "")
    if draft_model_env and draft_model_env != model_name:
        logger.warning(
            "Tokenizer mismatch: orchestrator uses '%s' but SPECSPLIT_DRAFT_MODEL_NAME='%s'. "
            "Speculative decoding requires identical vocabularies.",
            model_name,
            draft_model_env,
        )
    if target_model_env and target_model_env != model_name:
        logger.warning(
            "Tokenizer mismatch: orchestrator uses '%s' but SPECSPLIT_TARGET_MODEL_NAME='%s'. "
            "Speculative decoding requires identical vocabularies.",
            model_name,
            target_model_env,
        )

    orch = Orchestrator(config=config, model_name=model_name)
    orch.connect()
    output_text, result = orch.run_with_result_sync(args.prompt)

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

    if args.telemetry_output:
        orch.export_telemetry(args.telemetry_output)
        print(f"\nTelemetry exported to: {args.telemetry_output}")


if __name__ == "__main__":
    main()
