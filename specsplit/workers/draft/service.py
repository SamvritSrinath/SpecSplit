"""Draft Worker gRPC service bindings.

Exposes the ``DraftService`` gRPC server that wraps the ``DraftEngine``
for network-accessible speculative generation.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import Any

import grpc

from specsplit.core.config import DraftWorkerConfig
from specsplit.core.telemetry import TelemetryLogger
from specsplit.proto import spec_decoding_pb2, spec_decoding_pb2_grpc
from specsplit.workers.draft.engine import DraftEngine, TokenNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protobuf conversion helpers
# ---------------------------------------------------------------------------


def _to_proto_node(node: TokenNode) -> spec_decoding_pb2.TokenNode:
    """Recursively convert an engine ``TokenNode`` to its protobuf equivalent.

    Args:
        node: An in-memory ``TokenNode`` dataclass from the draft engine.

    Returns:
        A ``spec_decoding_pb2.TokenNode`` protobuf message.
    """
    return spec_decoding_pb2.TokenNode(
        token_id=node.token_id,
        log_prob=node.log_prob,
        children=[_to_proto_node(c) for c in node.children],
    )


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------


class DraftServiceServicer(spec_decoding_pb2_grpc.DraftServiceServicer):
    """gRPC servicer implementing the ``DraftService`` RPC interface.

    Each RPC call is wrapped with a telemetry span for distributed tracing.

    Args:
        engine: The draft generation engine.
        telemetry: Optional telemetry logger for span collection.
    """

    def __init__(
        self,
        engine: DraftEngine,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self._engine = engine
        self._telemetry = telemetry or TelemetryLogger(service_name="draft-worker")

    def GenerateDrafts(  # noqa: N802
        self,
        request: spec_decoding_pb2.DraftRequest,
        context: grpc.ServicerContext,
    ) -> spec_decoding_pb2.DraftResponse:
        """Handle a ``GenerateDrafts`` RPC call.

        Args:
            request: A ``DraftRequest`` protobuf message.
            context: gRPC server context.

        Returns:
            A ``DraftResponse`` protobuf message with the generated tree.
        """
        with self._telemetry.span(
            "generate_drafts",
            request_id=request.request_id,
            max_draft_len=request.max_draft_len,
        ):
            prompt_ids: list[int] = list(request.prompt_token_ids)
            roots: list[TokenNode] = self._engine.generate_draft_tree(
                prompt_ids=prompt_ids,
                k=request.max_draft_len or None,
                num_beams=request.num_beams or None,
                temperature=request.temperature or None,
            )

            response = spec_decoding_pb2.DraftResponse(
                request_id=request.request_id,
                draft_tree=[_to_proto_node(r) for r in roots],
            )

            logger.info(
                "GenerateDrafts completed: request_id=%s, roots=%d",
                request.request_id,
                len(roots),
            )
            return response

    def Ping(  # noqa: N802
        self,
        request: spec_decoding_pb2.PingRequest,
        context: grpc.ServicerContext,
    ) -> spec_decoding_pb2.PingResponse:
        """Health check endpoint."""
        logger.debug("Ping received")
        return spec_decoding_pb2.PingResponse(status="ok", worker_type="draft")


def serve(config: DraftWorkerConfig | None = None) -> None:
    """Start the Draft Worker gRPC server.

    Args:
        config: Optional configuration override. If ``None``, reads from
            environment variables.
    """
    config = config or DraftWorkerConfig()
    engine = DraftEngine(config=config)
    engine.load_model()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    servicer = DraftServiceServicer(engine)
    spec_decoding_pb2_grpc.add_DraftServiceServicer_to_server(servicer, server)

    bind_address = f"[::]:{config.grpc_port}"
    server.add_insecure_port(bind_address)
    server.start()
    logger.info("Draft Worker serving on %s", bind_address)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Draft Worker shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
