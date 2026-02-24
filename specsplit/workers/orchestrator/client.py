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
from typing import Any

import grpc

from specsplit.core.config import OrchestratorConfig
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
        """Establish gRPC channels to Draft and Target workers."""
        self._draft_channel = grpc.insecure_channel(self.config.draft_address)
        self._draft_stub = spec_decoding_pb2_grpc.DraftServiceStub(
            self._draft_channel,
        )

        self._target_channel = grpc.insecure_channel(self.config.target_address)
        self._target_stub = spec_decoding_pb2_grpc.TargetServiceStub(
            self._target_channel,
        )

        logger.info("gRPC channels established")

    def run_with_result(self, prompt: str) -> tuple[str, PipelineResult]:
        """Run the full speculative decoding pipeline for a given prompt.

        Tokenizes the prompt, executes the async speculative loop over
        gRPC, and decodes the resulting tokens back to a string.

        Args:
            prompt: The user's input text prompt.

        Returns:
            A tuple of (generated output text, PipelineResult with full metrics).
        """
        logger.info("Starting generation for prompt: %r", prompt[:80])

        tokenizer = self._ensure_tokenizer()
        prompt_ids: list[int] = tokenizer.encode(prompt)
        eos_token_id: int = tokenizer.eos_token_id or 2

        session_id = "" if not self.config.use_target_kv_cache else "default"
        with self._telemetry.span("full_pipeline", prompt_len=len(prompt)):
            result: PipelineResult = asyncio.run(
                run_speculative_loop_async(
                    draft_stub=self._draft_stub,
                    target_stub=self._target_stub,
                    prompt_ids=prompt_ids,
                    config=self.config,
                    session_id=session_id,
                    eos_token_id=eos_token_id,
                )
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

        return output_text, result

    def run(self, prompt: str) -> str:
        """Run the pipeline and return the generated text.

        Thin wrapper around :meth:`run_with_result` for callers that only
        need the output string.

        Args:
            prompt: The user's input text prompt.

        Returns:
            The generated output text.
        """
        output_text, _ = self.run_with_result(prompt)
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
        default="gpt2",
        help="HuggingFace model name for tokenizer (default: gpt2)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )

    config_kw: dict = {"use_target_kv_cache": args.use_target_cache}
    if args.max_rounds is not None:
        config_kw["max_rounds"] = args.max_rounds
    config = OrchestratorConfig(**config_kw)

    orch = Orchestrator(config=config, model_name=args.model_name)
    orch.connect()
    result = orch.run(args.prompt)

    print(f"\n{'=' * 60}")
    print("Generated Output:")
    print(f"{'=' * 60}")
    print(result)

    if args.telemetry_output:
        orch.export_telemetry(args.telemetry_output)
        print(f"\nTelemetry exported to: {args.telemetry_output}")


if __name__ == "__main__":
    main()
