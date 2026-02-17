"""Draft Worker gRPC service bindings.

Exposes the ``DraftService`` gRPC server that wraps the ``DraftEngine``
for network-accessible speculative generation.
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from specsplit.core.config import DraftWorkerConfig
from specsplit.core.telemetry import TelemetryLogger
from specsplit.workers.draft.engine import DraftEngine

logger = logging.getLogger(__name__)

# NOTE: The generated protobuf modules are imported at runtime after
# `make proto` has been run. The imports below will fail until then.
# from specsplit.proto import spec_decoding_pb2
# from specsplit.proto import spec_decoding_pb2_grpc


class DraftServiceServicer:
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

    def GenerateDrafts(self, request, context):  # noqa: N802
        """Handle a ``GenerateDrafts`` RPC call.

        Args:
            request: A ``DraftRequest`` protobuf message.
            context: gRPC server context.

        Returns:
            A ``DraftResponse`` protobuf message with the generated tree.

        .. todo::
            Convert ``TokenNode`` objects to protobuf and attach telemetry.
        """
        with self._telemetry.span(
            "generate_drafts",
            request_id=request.request_id,
            max_draft_len=request.max_draft_len,
        ):
            prompt_ids = list(request.prompt_token_ids)
            roots = self._engine.generate_draft_tree(
                prompt_ids=prompt_ids,
                k=request.max_draft_len or None,
                num_beams=request.num_beams or None,
                temperature=request.temperature or None,
            )

            # TODO(proto-conversion): Convert TokenNode → protobuf TokenNode
            # response = spec_decoding_pb2.DraftResponse(
            #     request_id=request.request_id,
            #     draft_tree=[_to_proto_node(r) for r in roots],
            # )
            # return response

            logger.info(
                "GenerateDrafts completed: request_id=%s, roots=%d",
                request.request_id,
                len(roots),
            )

    def Ping(self, request, context):  # noqa: N802
        """Health check endpoint.

        .. todo::
            Return a ``PingResponse`` protobuf with status="ok".
        """
        logger.debug("Ping received")
        # return spec_decoding_pb2.PingResponse(status="ok", worker_type="draft")


def serve(config: DraftWorkerConfig | None = None) -> None:
    """Start the Draft Worker gRPC server.

    Args:
        config: Optional configuration override. If ``None``, reads from
            environment variables.

    .. todo::
        Register the servicer with the generated gRPC server class.
    """
    config = config or DraftWorkerConfig()
    engine = DraftEngine(config=config)
    engine.load_model()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    # TODO(grpc-registration): Uncomment after `make proto`:
    # servicer = DraftServiceServicer(engine)
    # spec_decoding_pb2_grpc.add_DraftServiceServicer_to_server(servicer, server)

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
