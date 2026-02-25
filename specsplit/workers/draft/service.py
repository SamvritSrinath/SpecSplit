"""Draft Worker gRPC service bindings.

Exposes the ``DraftService`` gRPC server that wraps the ``DraftEngine``
for network-accessible speculative generation.
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from specsplit.core.config import DraftWorkerConfig
from specsplit.core.telemetry import Stopwatch, TelemetryLogger
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


def _count_nodes(node: TokenNode) -> int:
    """Count all descendant nodes (excluding self) in a TokenNode tree."""
    return sum(1 + _count_nodes(c) for c in node.children)


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
            # Issue 9: Invalidate draft KV cache on speculation miss
            if request.reset_cache:
                self._engine.reset_cache(
                    session_id=request.session_id or None,
                )
                logger.info("Draft KV cache reset (reset_cache=True)")

            prompt_ids: list[int] = list(request.prompt_token_ids)
            sw = Stopwatch()
            sw.start()
            # Use request values; proto default 0.0 for temperature.
            # Fix Bug 5: proto3 defaults unset float to 0.0, so request.temperature >= 0
            # is always true. Use temp > 0 to mean "use request"; else None → use
            # config.temperature (SPECSPLIT_DRAFT_TEMPERATURE env var).
            k = request.max_draft_len if request.max_draft_len > 0 else None
            num_beams = request.num_beams if request.num_beams > 0 else None
            temp = request.temperature if request.temperature > 0 else None

            roots: list[TokenNode] = self._engine.generate_draft_tree(
                prompt_ids=prompt_ids,
                k=k,
                num_beams=num_beams,
                temperature=temp,
                session_id=request.session_id or None,
            )
            sw.stop()

            # Issue 8: Populate TelemetryMetadata with server-side timing
            telemetry_meta = spec_decoding_pb2.TelemetryMetadata(
                span_id=request.request_id,
                wall_time_ms=sw.elapsed_ms,
                model_time_ms=sw.elapsed_ms,
                tokens_processed=sum(1 + _count_nodes(r) for r in roots),
                device=str(self._engine.device),
            )

            response = spec_decoding_pb2.DraftResponse(
                request_id=request.request_id,
                draft_tree=[_to_proto_node(r) for r in roots],
                telemetry=telemetry_meta,
            )

            effective_k = k or self._engine.config.max_draft_tokens
            effective_temp = temp if temp is not None else self._engine.config.temperature
            logger.info(
                "GenerateDrafts completed: request_id=%s, roots=%d, k=%d, temp=%.2f, model_ms=%.1f",
                request.request_id,
                len(roots),
                effective_k,
                effective_temp,
                sw.elapsed_ms,
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
