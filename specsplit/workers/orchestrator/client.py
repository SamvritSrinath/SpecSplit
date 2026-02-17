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
import logging
import sys
from typing import Any

from specsplit.core.config import OrchestratorConfig
from specsplit.core.telemetry import TelemetryLogger

logger = logging.getLogger(__name__)

# NOTE: Import generated protobuf modules after `make proto`:
# import grpc
# from specsplit.proto import spec_decoding_pb2
# from specsplit.proto import spec_decoding_pb2_grpc


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
    """

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.config = config or OrchestratorConfig()
        self._telemetry = TelemetryLogger(service_name="orchestrator")
        self._draft_stub: Any = None
        self._target_stub: Any = None

        logger.info(
            "Orchestrator initialized (draft=%s, target=%s)",
            self.config.draft_address,
            self.config.target_address,
        )

    def connect(self) -> None:
        """Establish gRPC channels to Draft and Target workers.

        .. todo::
            Create gRPC channels and stubs from generated code.
        """
        # TODO(grpc-channels): Uncomment after `make proto`:
        # draft_channel = grpc.insecure_channel(self.config.draft_address)
        # self._draft_stub = spec_decoding_pb2_grpc.DraftServiceStub(draft_channel)
        #
        # target_channel = grpc.insecure_channel(self.config.target_address)
        # self._target_stub = spec_decoding_pb2_grpc.TargetServiceStub(target_channel)
        logger.info("gRPC channels established (stubbed)")

    def run(self, prompt: str) -> str:
        """Run the full speculative decoding pipeline for a given prompt.

        Args:
            prompt: The user's input text prompt.

        Returns:
            The generated output text.

        .. todo::
            Implement the full tokenization → draft → verify → decode loop.
        """
        logger.info("Starting generation for prompt: %r", prompt[:80])

        with self._telemetry.span("full_pipeline", prompt_len=len(prompt)):
            # TODO(pipeline): Implement the actual pipeline:
            #
            # 1. Tokenize the prompt
            # 2. Loop for max_rounds:
            #    a. Call DraftService.GenerateDrafts
            #    b. Call TargetService.VerifyDrafts
            #    c. Append accepted tokens
            #    d. Check stopping criteria
            # 3. Decode final token sequence to text

            output_text = f"[STUB] Generated output for: {prompt[:40]}..."
            logger.info("Pipeline complete (stubbed)")

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
        "--telemetry-output",
        type=str,
        default=None,
        help="Path to export telemetry JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    )

    config = OrchestratorConfig()
    if args.max_rounds is not None:
        config = OrchestratorConfig(max_rounds=args.max_rounds)

    orch = Orchestrator(config=config)
    orch.connect()
    result = orch.run(args.prompt)

    print(f"\n{'='*60}")
    print("Generated Output:")
    print(f"{'='*60}")
    print(result)

    if args.telemetry_output:
        orch.export_telemetry(args.telemetry_output)
        print(f"\nTelemetry exported to: {args.telemetry_output}")


if __name__ == "__main__":
    main()
