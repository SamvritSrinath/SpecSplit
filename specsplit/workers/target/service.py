"""Target Worker gRPC service bindings.

Exposes the ``TargetService`` gRPC server that wraps the ``TargetEngine``
for network-accessible tree-attention verification with session-based
KV caching.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import Any

import grpc

from specsplit.core.config import TargetWorkerConfig
from specsplit.core.telemetry import Stopwatch, TelemetryLogger
from specsplit.proto import spec_decoding_pb2, spec_decoding_pb2_grpc
from specsplit.workers.target.engine import TargetEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protobuf conversion helpers
# ---------------------------------------------------------------------------


def _from_proto_node(node: spec_decoding_pb2.TokenNode) -> dict[str, Any]:
    """Recursively convert a protobuf ``TokenNode`` to a plain Python dict.

    The dict format matches what ``TargetEngine.verify_draft_tree`` expects:
    ``{"token_id": int, "log_prob": float, "children": [...]}``.

    Args:
        node: A protobuf ``TokenNode`` message.

    Returns:
        A plain dict representation of the node and its descendants.
    """
    return {
        "token_id": node.token_id,
        "log_prob": node.log_prob,
        "children": [_from_proto_node(c) for c in node.children],
    }


def _count_tree_nodes(roots: list[dict[str, Any]]) -> int:
    """Count total number of nodes in a tree represented as list of root dicts."""
    return sum(
        1 + _count_tree_nodes(n.get("children", [])) for n in roots
    )


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------


class TargetServiceServicer(spec_decoding_pb2_grpc.TargetServiceServicer):
    """gRPC servicer implementing the ``TargetService`` RPC interface.

    Handles ``VerifyDrafts`` (with session-based KV caching) and
    ``EndSession`` (for explicit cache cleanup).

    Args:
        engine: The target verification engine.
        config: Target worker config (for request size limits).
        telemetry: Optional telemetry logger for span collection.
    """

    def __init__(
        self,
        engine: TargetEngine,
        config: TargetWorkerConfig | None = None,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self._engine = engine
        self._config = config or TargetWorkerConfig()
        self._telemetry = telemetry or TelemetryLogger(service_name="target-worker")

    def VerifyDrafts(  # noqa: N802
        self,
        request: spec_decoding_pb2.VerifyRequest,
        context: grpc.ServicerContext,
    ) -> spec_decoding_pb2.VerifyResponse:
        """Handle a ``VerifyDrafts`` RPC call with optional session KV caching.

        If ``request.session_id`` is non-empty, the engine reuses (or creates)
        a KV cache for that session, and automatically rolls it back to the
        accepted prefix after verification.

        Args:
            request: A ``VerifyRequest`` protobuf message.
            context: gRPC server context.

        Returns:
            A ``VerifyResponse`` protobuf message with verification results.
        """
        session_id: str | None = request.session_id or None

        with self._telemetry.span(
            "verify_drafts",
            request_id=request.request_id,
            session_id=session_id or "stateless",
        ):
            prompt_ids: list[int] = list(request.prompt_token_ids)
            if len(prompt_ids) > self._config.max_prompt_tokens:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"prompt length {len(prompt_ids)} exceeds max_prompt_tokens ({self._config.max_prompt_tokens})",
                )
            draft_tree: list[dict[str, Any]] = [_from_proto_node(n) for n in request.draft_tree]
            num_nodes = _count_tree_nodes(draft_tree)
            if num_nodes > self._config.max_tree_nodes:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"draft tree has {num_nodes} nodes, exceeds max_tree_nodes ({self._config.max_tree_nodes})",
                )

            sw = Stopwatch()
            sw.start()
            result = self._engine.verify_draft_tree(
                prompt_ids,
                draft_tree,
                session_id=session_id,
                temperature=request.temperature,
            )
            sw.stop()

            # Issue 8: Populate TelemetryMetadata with server-side timing
            telemetry_meta = spec_decoding_pb2.TelemetryMetadata(
                span_id=request.request_id,
                wall_time_ms=sw.elapsed_ms,
                model_time_ms=sw.elapsed_ms,
                tokens_processed=num_nodes,
                device=str(self._engine.device),
            )

            # Issue 17 / 30: We must avoid conflating "no correction" (None -> 0)
            # with "correction token is 0". We will add a has_correction flag to proto.
            has_corr = result.correction_token_id is not None
            
            response = spec_decoding_pb2.VerifyResponse(
                request_id=request.request_id,
                accepted_token_ids=result.accepted_token_ids,
                correction_token_id=result.correction_token_id if has_corr else 0,
                has_correction=has_corr,
                num_accepted=result.num_accepted,
                session_id=session_id or "",
                cache_hit=result.cache_hit,
                telemetry=telemetry_meta,
            )

            logger.info(
                "VerifyDrafts completed: request_id=%s, session=%s, cache_hit=%s, "
                "accepted=%d, model_ms=%.1f",
                request.request_id,
                session_id or "none",
                result.cache_hit,
                result.num_accepted,
                sw.elapsed_ms,
            )
            return response

    def EndSession(  # noqa: N802
        self,
        request: spec_decoding_pb2.EndSessionRequest,
        context: grpc.ServicerContext,
    ) -> spec_decoding_pb2.EndSessionResponse:
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
            was_active: bool = self._engine.end_session(request.session_id)

            response = spec_decoding_pb2.EndSessionResponse(
                session_id=request.session_id,
                was_active=was_active,
            )

            logger.info(
                "EndSession: session=%s, was_active=%s",
                request.session_id,
                was_active,
            )
            return response

    def Ping(  # noqa: N802
        self,
        request: spec_decoding_pb2.PingRequest,
        context: grpc.ServicerContext,
    ) -> spec_decoding_pb2.PingResponse:
        """Health check endpoint."""
        logger.debug("Ping received (active_sessions=%d)", self._engine.active_sessions)
        return spec_decoding_pb2.PingResponse(status="ok", worker_type="target")


def serve(config: TargetWorkerConfig | None = None) -> None:
    """Start the Target Worker gRPC server.

    Args:
        config: Optional configuration override.
    """
    config = config or TargetWorkerConfig()
    engine = TargetEngine(config=config)
    engine.load_model()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    servicer = TargetServiceServicer(engine, config=config)
    spec_decoding_pb2_grpc.add_TargetServiceServicer_to_server(servicer, server)

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
