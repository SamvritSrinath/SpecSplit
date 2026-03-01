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
        self._loop: asyncio.AbstractEventLoop | None = None
        self._draft_stub: spec_decoding_pb2_grpc.DraftServiceStub | None = None
        self._target_stub: spec_decoding_pb2_grpc.TargetServiceStub | None = None
        self._tokenizer: Any = None
        self._vocab_bridge: Any | None = None
        self._sync_executor: concurrent.futures.ThreadPoolExecutor | None = None

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
        
        Also validates vocabulary alignment between draft/target models if they differ.
        If strict_vocab_check is False, it initializes the VocabBridge here.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info("Tokenizer loaded: %s", self.model_name)
            
            # Check draft and target vocabularies
            draft_model_env = os.environ.get("SPECSPLIT_DRAFT_MODEL_NAME", "")
            target_model_env = os.environ.get("SPECSPLIT_TARGET_MODEL_NAME", "")
            
            if draft_model_env and target_model_env and draft_model_env != target_model_env:
                logger.info("Draft and Target use different models. Checking vocabulary alignment...")
                draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_env)
                target_tokenizer = AutoTokenizer.from_pretrained(target_model_env)
                
                if len(draft_tokenizer) != len(target_tokenizer):
                    if self.config.strict_vocab_check:
                        msg = (
                            f"Vocabulary mismatch: Draft vocab size {len(draft_tokenizer)}, "
                            f"Target vocab size {len(target_tokenizer)}. "
                            "With strict_vocab_check=True, heterogeneous models must "
                            "have the same vocabulary size. Set strict_vocab_check=False "
                            "to enable VocabBridge."
                        )
                        logger.error(msg)
                        raise RuntimeError(msg)
                    else:
                        from specsplit.workers.orchestrator.vocab_bridge import VocabBridge
                        self._vocab_bridge = VocabBridge(draft_tokenizer, target_tokenizer)
                        logger.info("strict_vocab_check is False. Initialized VocabBridge.")

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

        # Create gRPC channels in the current event loop (grpc.aio channels are
        # loop-bound; creating them from sync code causes "attached to different loop")
        if self._draft_channel is None:
            self.connect()

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
                    vocab_bridge=self._vocab_bridge,
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
                    if inspect.isawaitable(end_resp):
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

    def export_telemetry(self, path: str) -> None:
        """Export collected telemetry spans to a JSON file."""
        self._telemetry.export(path)


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
            try:
                from specsplit.proto import spec_decoding_pb2

                end_req = spec_decoding_pb2.EndSessionRequest(session_id=self.session_id)
                end_req = spec_decoding_pb2.EndSessionRequest(session_id=self.session_id)
                coro = self.orchestrator._target_stub.EndSession(end_req)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(coro)
                except RuntimeError:
                    asyncio.run(coro)
                logger.debug("ConversationSession %s ended.", self.session_id)
            except Exception:
                logger.debug("ConversationSession %s cleanup failed (non-critical).", self.session_id)


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
    if args.max_draft_tokens is not None:
        config_kw["max_draft_tokens"] = args.max_draft_tokens
    if args.draft_temperature is not None:
        config_kw["draft_temperature"] = args.draft_temperature
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

    if args.telemetry_output:
        orch.export_telemetry(args.telemetry_output)
        print(f"\nTelemetry exported to: {args.telemetry_output}")


if __name__ == "__main__":
    main()
