"""Target Worker gRPC service bindings.

Exposes the ``TargetService`` gRPC server that wraps the ``TargetEngine``
for network-accessible tree-attention verification with session-based
KV caching.
"""

from __future__ import annotations

import logging
from collections import deque
from concurrent import futures
from typing import Any

import grpc
import torch

from specsplit.core.config import TargetWorkerConfig
from specsplit.core.telemetry import Stopwatch, TelemetryLogger
from specsplit.proto import spec_decoding_pb2, spec_decoding_pb2_grpc
from specsplit.workers.target.engine import CacheDesyncError, TargetEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protobuf conversion helpers
# ---------------------------------------------------------------------------


def _decode_proto_tree_direct(
    proto_nodes: list[spec_decoding_pb2.TokenNode],
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], list[int], list[float], torch.Tensor | None]:
    """Single-pass BFS: proto -> flat arrays. No intermediate dicts."""
    token_ids: list[int] = []
    topology_map: list[int] = []
    log_probs: list[float] = []
    all_indices: list[list[int]] = []
    all_values: list[float] = []
    has_top_k = False

    queue: deque[tuple[spec_decoding_pb2.TokenNode, int]] = deque()
    for root in proto_nodes:
        queue.append((root, -1))

    while queue:
        node, parent_idx = queue.popleft()
        current_idx = len(token_ids)
        token_ids.append(node.token_id)
        topology_map.append(parent_idx)
        log_probs.append(node.log_prob)

        if node.top_k_token_ids and node.top_k_probs:
            has_top_k = True
            for tid, p in zip(node.top_k_token_ids, node.top_k_probs):
                if 0 <= tid < vocab_size:
                    all_indices.append([current_idx, tid])
                    all_values.append(float(p))

        for child in node.children:
            queue.append((child, current_idx))

    draft_probs_full = None
    if has_top_k and all_indices:
        draft_probs_full = torch.zeros((len(token_ids), vocab_size), dtype=dtype, device=device)
        idx_t = torch.tensor(all_indices, dtype=torch.long, device=device)
        val_t = torch.tensor(all_values, dtype=dtype, device=device)
        draft_probs_full[idx_t[:, 0], idx_t[:, 1]] = val_t

    return token_ids, topology_map, log_probs, draft_probs_full


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
            new_ids: list[int] = list(getattr(request, "new_token_ids", []))

            # Bug 3: Delta-only mode — when prompt_token_ids is empty but
            # new_token_ids is populated, the orchestrator is sending only
            # the delta. The target engine's session cache already has the
            # prefix and only needs the new tokens appended.
            #
            # CRITICAL GUARD: If the session was evicted (TTL or LRU), the
            # engine no longer has the prefix. Using only the 2-3 delta
            # tokens as the full prompt would cause the 70B model to
            # verify against a nonsensical context, producing garbage.
            if not prompt_ids and new_ids:
                with self._engine._session_dict_lock:
                    if not session_id or session_id not in self._engine._session_caches:
                        context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "CACHE_EVICTED_DELTA_ONLY: session cache was evicted; "
                            "orchestrator must retry with full context",
                        )
                prompt_ids = new_ids

            if prompt_ids and len(prompt_ids) > self._config.max_prompt_tokens:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"prompt length {len(prompt_ids)} exceeds max_prompt_tokens ({self._config.max_prompt_tokens})",
                )

            sw = Stopwatch()
            sw.start()

            vocab_size = self._engine._model.config.vocab_size if self._engine._is_loaded else 32000
            device = self._engine.device
            dtype = self._engine._model.dtype if self._engine._is_loaded else torch.float16

            payload_bytes = request.ByteSize()
            with self._telemetry.span("proto_decode") as decode_span:
                flat_token_ids, topology_map, flat_log_probs, flat_draft_probs_full = _decode_proto_tree_direct(
                    request.draft_tree, vocab_size, device, dtype
                )
            proto_decode_ms = decode_span.elapsed_ms

            num_nodes = len(flat_token_ids)
            if num_nodes > self._config.max_tree_nodes:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"draft tree has {num_nodes} nodes, exceeds max_tree_nodes ({self._config.max_tree_nodes})",
                )

            expected_prefix_length = int(getattr(request, "expected_prefix_length", 0) or 0)

            try:
                result = self._engine.verify_draft_tree(
                    prompt_ids,
                    flat_token_ids,
                    topology_map,
                    flat_log_probs,
                    flat_draft_probs_full,
                    session_id=session_id,
                    temperature=request.temperature,
                    expected_prefix_length=expected_prefix_length,
                )
            except CacheDesyncError as e:
                sw.stop()
                logger.warning("CacheDesyncError in engine: %s", str(e))
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    str(e),
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

            # Store P-1 specific telemetry in span context metadata automatically
            # Not added to protobuf to preserve schema, but injected into telemetry backend
            import specsplit.core.telemetry as telemetry
            ctx = telemetry.get_current_context()
            if ctx is not None:
                ctx.metadata["proto_decode_ms"] = proto_decode_ms
                ctx.metadata["payload_bytes"] = payload_bytes

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
