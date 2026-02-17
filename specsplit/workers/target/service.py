"""Target Worker gRPC service bindings.

Exposes the ``TargetService`` gRPC server that wraps the ``TargetEngine``
for network-accessible tree-attention verification with session-based
KV caching.
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from specsplit.core.config import TargetWorkerConfig
from specsplit.core.telemetry import TelemetryLogger
from specsplit.workers.target.engine import TargetEngine

logger = logging.getLogger(__name__)

# NOTE: Import generated protobuf modules after `make proto`:
# from specsplit.proto import spec_decoding_pb2
# from specsplit.proto import spec_decoding_pb2_grpc


class TargetServiceServicer:
    """gRPC servicer implementing the ``TargetService`` RPC interface.

    Handles ``VerifyDrafts`` (with session-based KV caching) and
    ``EndSession`` (for explicit cache cleanup).

    Args:
        engine: The target verification engine.
        telemetry: Optional telemetry logger for span collection.
    """

    def __init__(
        self,
        engine: TargetEngine,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self._engine = engine
        self._telemetry = telemetry or TelemetryLogger(service_name="target-worker")

    def VerifyDrafts(self, request, context):  # noqa: N802
        """Handle a ``VerifyDrafts`` RPC call with optional session KV caching.

        If ``request.session_id`` is non-empty, the engine reuses (or creates)
        a KV cache for that session, and automatically rolls it back to the
        accepted prefix after verification.

        Args:
            request: A ``VerifyRequest`` protobuf message.
            context: gRPC server context.

        Returns:
            A ``VerifyResponse`` protobuf message with verification results.

        .. todo::
            Convert protobuf tree to dict, call engine, convert result back.
        """
        session_id = request.session_id or None

        with self._telemetry.span(
            "verify_drafts",
            request_id=request.request_id,
            session_id=session_id or "stateless",
        ):
            prompt_ids = list(request.prompt_token_ids)

            # TODO(proto-conversion): Convert proto TokenNode → dict tree
            # draft_tree = [_from_proto_node(n) for n in request.draft_tree]
            draft_tree: list = []  # Placeholder

            result = self._engine.verify_draft_tree(
                prompt_ids,
                draft_tree,
                session_id=session_id,
            )

            # TODO(proto-conversion): Build and return VerifyResponse
            # response = spec_decoding_pb2.VerifyResponse(
            #     request_id=request.request_id,
            #     accepted_token_ids=result.accepted_token_ids,
            #     correction_token_id=result.correction_token_id or 0,
            #     num_accepted=result.num_accepted,
            #     session_id=session_id or "",
            #     cache_hit=result.cache_hit,
            # )
            # return response

            logger.info(
                "VerifyDrafts completed: request_id=%s, session=%s, "
                "cache_hit=%s, accepted=%d",
                request.request_id,
                session_id or "none",
                result.cache_hit,
                result.num_accepted,
            )

    def EndSession(self, request, context):  # noqa: N802
        """Handle an ``EndSession`` RPC — release a session's KV cache.

        Args:
            request: An ``EndSessionRequest`` protobuf message.
            context: gRPC server context.

        Returns:
            An ``EndSessionResponse`` protobuf message.
        """
        with self._telemetry.span(
            "end_session",
            session_id=request.session_id,
        ):
            was_active = self._engine.end_session(request.session_id)

            # TODO(proto-conversion): Build and return EndSessionResponse
            # response = spec_decoding_pb2.EndSessionResponse(
            #     session_id=request.session_id,
            #     was_active=was_active,
            # )
            # return response

            logger.info(
                "EndSession: session=%s, was_active=%s",
                request.session_id,
                was_active,
            )

    def Ping(self, request, context):  # noqa: N802
        """Health check endpoint."""
        logger.debug("Ping received (active_sessions=%d)", self._engine.active_sessions)
        # return spec_decoding_pb2.PingResponse(status="ok", worker_type="target")


def serve(config: TargetWorkerConfig | None = None) -> None:
    """Start the Target Worker gRPC server.

    Args:
        config: Optional configuration override.

    .. todo::
        Register the servicer with the generated gRPC server class.
    """
    config = config or TargetWorkerConfig()
    engine = TargetEngine(config=config)
    engine.load_model()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    # TODO(grpc-registration): Uncomment after `make proto`:
    # servicer = TargetServiceServicer(engine)
    # spec_decoding_pb2_grpc.add_TargetServiceServicer_to_server(servicer, server)

    bind_address = f"[::]:{config.grpc_port}"
    server.add_insecure_port(bind_address)
    server.start()
    logger.info(
        "Target Worker serving on %s (max_sessions=%d)",
        bind_address,
        config.max_sessions,
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Target Worker shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
