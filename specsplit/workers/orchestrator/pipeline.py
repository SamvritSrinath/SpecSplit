"""Overlapped Async Orchestrator for Disaggregated Speculative Decoding.

The key insight of speculative decoding is that latency is dominated by
network round-trips (Draft → Orchestrator → Target → Orchestrator) and
the Target model's forward pass.  This module hides that latency by
**overlapping** operations:

    While TargetService.VerifyDrafts(Tree N) is in flight:
        → Speculatively fire DraftService.GenerateDrafts(Tree N+1)
          assuming the longest branch of Tree N will be accepted.

If the assumption is correct (high acceptance), Tree N+1 is ready
immediately.  If the assumption is wrong (rejection), Tree N+1 is
discarded and we re-draft from the corrected context.

Pipeline Architecture
---------------------
::

    ┌─────────────┐    Tree N    ┌─────────────┐
    │ Orchestrator │─────────────▶│ Target Svc  │  ← VerifyDrafts(N)
    │              │             │              │
    │  (async)     │  Tree N+1   ┌─────────────┐
    │              │─────────────▶│  Draft Svc  │  ← GenerateDrafts(N+1)
    │              │             │  (specul.)   │     (runs concurrently!)
    └──────────────┘             └──────────────┘

    Time ──▶
    │ verify(N) ████████████████████│
    │ draft(N+1)    ████████│       │  ← overlapped!
    │                       │   verify(N+1) ██████████████│
    │                       │   draft(N+2)      ██████│   │


The ``draft(N+1)`` call runs *during* ``verify(N)``, saving one full
round-trip per iteration when the speculation is correct.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import grpc

from specsplit.core.config import OrchestratorConfig
from specsplit.core.telemetry import Stopwatch, TelemetryLogger
from specsplit.proto import spec_decoding_pb2

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _RpcResult(Generic[T]):
    """Wrapper pairing an RPC response with its wall-clock duration."""

    value: T
    elapsed_ms: float


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DraftTree:
    """Represents a speculative draft tree produced by the Draft Worker.

    Attributes:
        token_ids: Flat list of drafted token IDs in tree order.
        topology_map: Parent indices for each node (-1 = root).
        proto_nodes: The raw protobuf ``TokenNode`` messages, preserved
            for direct forwarding to the Target Worker without re-encoding.
        round_idx: The pipeline round that produced this tree.
    """

    token_ids: list[int]
    topology_map: list[int]
    proto_nodes: list[Any] = field(default_factory=list)
    round_idx: int = 0


@dataclass
class VerificationResult:
    """Result of a VerifyDrafts RPC call.

    Attributes:
        accepted_tokens: List of accepted token IDs along the best path.
        bonus_token: The target model's correction/continuation token.
        accepted_length: Number of accepted draft tokens.
        cache_hit: Whether the target reused KV cache.
        round_idx: The pipeline round this corresponds to.
    """

    accepted_tokens: list[int]
    accepted_length: int
    bonus_token: int | None = None
    cache_hit: bool = False
    round_idx: int = 0


@dataclass
class PIDState:
    integral: float = 0.0
    prev_error: float = 0.0
    kp: float = 2.0
    ki: float = 0.5
    kd: float = 0.1
    setpoint: float = 0.7  # Target acceptance rate


@dataclass
class RoundMetrics:
    """Per-round metrics for fine-grained acceptance analysis."""

    round_idx: int
    accepted: int
    path_depth: int
    tree_nodes: int
    acceptance_rate: float


@dataclass
class SpeculativeState:
    """Tracks the current state of the speculative pipeline.

    Attributes:
        generated_tokens: All tokens generated so far (accepted + bonus).
        prompt_ids: The original prompt token IDs.
        total_rounds: Number of verify rounds completed.
        total_accepted: Total draft tokens accepted across all rounds.
        total_path_depth: Sum of longest-path depths across all rounds
            (denominator for acceptance rate).
        total_tree_nodes: Total draft tree nodes generated (for diagnostics).
        speculation_hits: Number of times N+1 speculation was correct.
        speculation_misses: Number of times N+1 was discarded.
        is_finished: Whether generation has reached a stop condition.
    """

    generated_tokens: list[int] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    total_rounds: int = 0
    total_accepted: int = 0
    total_path_depth: int = 0
    total_tree_nodes: int = 0
    speculation_hits: int = 0
    speculation_misses: int = 0
    total_rpc_time_ms: float = 0.0
    per_round_metrics: list[RoundMetrics] = field(default_factory=list)
    is_finished: bool = False
    
    # PR-7 / PR-8: Adaptive K and Fallback Mode State
    current_k: int = 0  # Initialized from config.max_draft_tokens
    recent_acceptance_rates: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    is_fallback_mode: bool = False
    consecutive_low_acceptance: int = 0
    pid: PIDState = field(default_factory=PIDState)


@dataclass
class PipelineResult:
    """Final result of the speculative decoding pipeline.

    Attributes:
        output_tokens: The generated token IDs (excluding the prompt).
        total_rounds: Number of draft→verify rounds executed.
        acceptance_rate: Overall acceptance rate across all rounds.
        speculation_hit_rate: Fraction of N+1 speculations that were correct.
        wall_time_ms: Total wall-clock time.
        telemetry: Collected telemetry spans.
    """

    output_tokens: list[int]
    total_rounds: int
    acceptance_rate: float
    speculation_hit_rate: float
    wall_time_ms: float
    network_idle_ms: float = 0.0
    per_round_acceptance: list[dict[str, Any]] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Protobuf ↔ DraftTree Conversion Helpers
# ============================================================================


def _flatten_proto_tree(proto_nodes: list[Any]) -> tuple[list[int], list[int]]:
    """Flatten nested protobuf ``TokenNode`` messages into parallel arrays.

    Performs a DFS traversal and assigns contiguous indices.

    Args:
        proto_nodes: Root-level ``spec_decoding_pb2.TokenNode`` messages.

    Returns:
        ``(token_ids, topology_map)`` where ``topology_map[i]`` is the
        parent index of node ``i`` (``-1`` for roots).
    """
    token_ids: list[int] = []
    topology_map: list[int] = []

    # Use BFS (deque) to match the target engine's _flatten_tree ordering.
    # Both sides MUST use the same traversal order so that node indices align
    # for branching trees (DFS and BFS diverge when num_beams > 1).
    queue: deque[tuple[Any, int]] = deque()
    for root in proto_nodes:
        queue.append((root, -1))

    while queue:
        node, parent_idx = queue.popleft()
        my_idx = len(token_ids)
        token_ids.append(node.token_id)
        topology_map.append(parent_idx)
        for child in node.children:
            queue.append((child, my_idx))

    return token_ids, topology_map


def _flatten_proto_tree_with_topk(
    proto_nodes: list[Any],
) -> tuple[list[int], list[int], list[tuple[list[int], list[float]]], list[float]]:
    """Flatten nested protobuf TokenNode messages with top-k draft distributions.

    Same BFS ordering as _flatten_proto_tree. Returns per-node top_k data
    and log_probs for correct residual sampling in stochastic verification.

    Returns:
        (token_ids, topology_map, top_k_per_node, log_probs) where:
        - top_k_per_node[i] is (top_k_token_ids, top_k_probs) for node i
        - log_probs[i] is the draft log-probability for node i (critical for
          stochastic acceptance: q_val = exp(log_prob); p_val >= q_val check).
    """
    token_ids: list[int] = []
    topology_map: list[int] = []
    top_k_per_node: list[tuple[list[int], list[float]]] = []
    log_probs: list[float] = []

    queue: deque[tuple[Any, int]] = deque()
    for root in proto_nodes:
        queue.append((root, -1))

    while queue:
        node, parent_idx = queue.popleft()
        my_idx = len(token_ids)
        token_ids.append(node.token_id)
        topology_map.append(parent_idx)
        tk_ids = list(node.top_k_token_ids) if node.top_k_token_ids else []
        tk_probs = list(node.top_k_probs) if node.top_k_probs else []
        top_k_per_node.append((tk_ids, tk_probs))
        log_probs.append(getattr(node, "log_prob", 0.0))
        for child in node.children:
            queue.append((child, my_idx))

    return token_ids, topology_map, top_k_per_node, log_probs


def _arrays_to_proto_nodes(
    token_ids: list[int],
    topology_map: list[int],
    top_k_per_node: list[tuple[list[int], list[float]]] | None = None,
    log_probs: list[float] | None = None,
) -> list[Any]:
    """Build protobuf TokenNode list from flat token_ids and topology_map.

    Inverse of _flatten_proto_tree. Uses BFS ordering to match the target
    engine's _flatten_tree. When log_probs is provided, each node gets the
    corresponding draft log-probability (required for stochastic verification:
    q_val = exp(log_prob); acceptance check p_val >= q_val). When top_k_per_node
    is provided, includes top-k distributions for correct residual sampling.

    Args:
        token_ids: Flat list of token IDs in BFS order.
        topology_map: Parent index for each node (-1 = root).
        top_k_per_node: Optional list of (top_k_token_ids, top_k_probs) per
            node in same order as token_ids. When absent, nodes get empty top-k.
        log_probs: Optional list of draft log-probabilities per node. When
            absent, nodes get log_prob=0.0 (causes q_val=1.0, breaks stochastic).

    Returns:
        Root-level spec_decoding_pb2.TokenNode protobuf messages.
    """
    from specsplit.proto import spec_decoding_pb2

    num_nodes = len(token_ids)
    if num_nodes == 0:
        return []

    if top_k_per_node is None:
        top_k_per_node = [([], []) for _ in range(num_nodes)]
    elif len(top_k_per_node) != num_nodes:
        top_k_per_node = top_k_per_node[:num_nodes] + [([], [])] * max(
            0, num_nodes - len(top_k_per_node)
        )

    if log_probs is None or len(log_probs) != num_nodes:
        log_probs_arr = [0.0] * num_nodes
    else:
        log_probs_arr = log_probs[:num_nodes]

    children: dict[int, list[int]] = {}
    for i, p in enumerate(topology_map):
        children.setdefault(p, []).append(i)
    roots = children.get(-1, [])

    def build_node(idx: int) -> Any:
        tk_ids, tk_probs = top_k_per_node[idx]
        return spec_decoding_pb2.TokenNode(
            token_id=token_ids[idx],
            log_prob=log_probs_arr[idx],
            children=[build_node(c) for c in children.get(idx, [])],
            top_k_token_ids=tk_ids,
            top_k_probs=tk_probs,
        )

    return [build_node(r) for r in roots]


def _get_longest_path(draft: DraftTree) -> list[int]:
    """Extract the longest root-to-leaf path from a flattened tree.

    When multiple paths have the same maximum length, the first such path
    encountered in DFS traversal order (stack-based) is returned.
    """
    children: dict[int, list[int]] = {}
    for i, p in enumerate(draft.topology_map):
        children.setdefault(p, []).append(i)

    best_path: list[int] = []
    stack = [(r, [draft.token_ids[r]]) for r in children.get(-1, [])]

    while stack:
        node, path = stack.pop()
        if len(path) > len(best_path):
            best_path = path
        for c in children.get(node, []):
            stack.append((c, path + [draft.token_ids[c]]))

    return best_path


# ============================================================================
# Async gRPC Stub Wrappers — Real gRPC calls
# ============================================================================


async def _call_generate_drafts(
    draft_stub: Any,
    prompt_ids: list[int],
    context_ids: list[int],
    config: OrchestratorConfig,
    round_idx: int,
    draft_len: int,
    reset_cache: bool = False,
    session_id: str | None = None,
) -> _RpcResult[DraftTree]:
    """Call ``DraftService.GenerateDrafts`` over gRPC.

    Constructs a ``DraftRequest`` protobuf, sends it to the Draft Worker,
    and converts the ``DraftResponse`` into a :class:`DraftTree`.

    Args:
        draft_stub: A gRPC stub for DraftService (sync or async).
        prompt_ids: Original prompt token IDs.
        context_ids: Current context (prompt + generated so far).
        config: Pipeline configuration.
        round_idx: Current round index (for tracing).
        draft_len: Dynamic depth of the speculative draft tree (Adaptive K).
        reset_cache: If True, instruct the draft engine to clear its
            KV cache before generating (e.g. after speculation miss).
        session_id: Session ID for per-session KV cache isolation
            on the Draft Worker.

    Returns:
        An :class:`_RpcResult` wrapping a :class:`DraftTree` and the RPC
        wall-clock elapsed time in milliseconds.

    Raises:
        grpc.aio.AioRpcError: Re-raised for non-timeout gRPC errors.
    """
    rpc_sw = Stopwatch()
    rpc_sw.start()

    # Inject Synthetic RTT Latency (Task 4.1) — inside stopwatch so elapsed_ms
    # includes it; avoids double-counting when accumulating total_rpc_time_ms.
    if config.simulated_rtt_ms > 0:
        await asyncio.sleep(config.simulated_rtt_ms / 1000.0)

    request = spec_decoding_pb2.DraftRequest(
        request_id=f"round-{round_idx}-{uuid.uuid4().hex[:8]}",
        prompt_token_ids=context_ids,
        max_draft_len=draft_len,
        temperature=config.draft_temperature,
        reset_cache=reset_cache,
        session_id=session_id or "",
    )

    # Issue 9: Pass timeout to prevent indefinite hangs
    response = draft_stub.GenerateDrafts(request, timeout=config.timeout_s)
    # Support both sync and async stubs (UnaryUnaryCall is awaitable but not a coroutine/future)
    if inspect.isawaitable(response):
        response = await response

    rpc_sw.stop()

    # Convert the nested proto tree to flat arrays
    proto_nodes = list(response.draft_tree)
    token_ids, topology_map = _flatten_proto_tree(proto_nodes)

    logger.debug(
        "Draft RPC completed: round=%d, tokens=%d, rpc_ms=%.2f",
        round_idx,
        len(token_ids),
        rpc_sw.elapsed_ms,
    )
    tree = DraftTree(
        token_ids=token_ids,
        topology_map=topology_map,
        proto_nodes=proto_nodes,
        round_idx=round_idx,
    )
    return _RpcResult(value=tree, elapsed_ms=rpc_sw.elapsed_ms)


async def _call_verify_drafts(
    target_stub: Any,
    context_ids: list[int],
    draft_tree: DraftTree,
    session_id: str | None,
    config: OrchestratorConfig,
    new_token_ids: list[int] | None = None,
    expected_prefix_length: int = 0,
    vocab_bridge: Any | None = None,
) -> _RpcResult[VerificationResult]:
    """Call ``TargetService.VerifyDrafts`` over gRPC.

    Forwards the draft tree's proto nodes directly to the Target Worker
    to avoid re-encoding.

    Args:
        target_stub: A gRPC stub for TargetService (sync or async).
        context_ids: Full current context (prompt + generated so far).
        draft_tree: The draft tree to verify (with preserved proto nodes).
        session_id: Session ID for KV cache reuse, or None for stateless.
        config: Pipeline configuration.
        new_token_ids: Delta tokens since last verified position (Issue 11).
            When provided, sent instead of full context to reduce bandwidth.
        vocab_bridge: Optional bridge to convert between draft/target vocabularies.

    Returns:
        An :class:`_RpcResult` wrapping a :class:`VerificationResult` and
        the RPC wall-clock elapsed time in milliseconds.

    Raises:
        grpc.aio.AioRpcError: Re-raised for non-timeout gRPC errors.
    """
    rpc_sw = Stopwatch()
    rpc_sw.start()

    # Inject Synthetic RTT Latency (Task 4.1) — inside stopwatch so elapsed_ms
    # includes it; avoids double-counting when accumulating total_rpc_time_ms.
    if config.simulated_rtt_ms > 0:
        await asyncio.sleep(config.simulated_rtt_ms / 1000.0)

    # Apply vocab_bridge translation if configured
    target_context_ids = context_ids
    target_new_token_ids = new_token_ids or []
    target_expected_prefix_length = expected_prefix_length
    
    # We omit translating draft_tree.proto_nodes as that would require mutating
    # the protobuf tree deeply. The VocabBridge primarily handles linear sequences.
    # To fully support heterogeneous trees, _arrays_to_proto_nodes should use it.
    
    request = spec_decoding_pb2.VerifyRequest(
        request_id=f"verify-{draft_tree.round_idx}-{uuid.uuid4().hex[:8]}",
        prompt_token_ids=target_context_ids,
        draft_tree=draft_tree.proto_nodes,
        session_id=session_id or "",
        temperature=config.verify_temperature,
        new_token_ids=target_new_token_ids,
        expected_prefix_length=target_expected_prefix_length,
    )

    # Issue 9: Pass timeout to prevent indefinite hangs
    response = target_stub.VerifyDrafts(request, timeout=config.timeout_s)
    # Support both sync and async stubs (UnaryUnaryCall is awaitable but not a coroutine/future)
    if inspect.isawaitable(response):
        response = await response

    rpc_sw.stop()

    accepted_tokens = list(response.accepted_token_ids)

    # Issue 39/40: Use has_correction proto field to correctly distinguish
    # between "no correction" and "correction token is 0".
    # proto3 defaults correction_token_id to 0 when unset, so we cannot
    # rely on the value alone.
    bonus = response.correction_token_id if response.has_correction else None

    # Apply vocab map backwards (target -> draft)
    if vocab_bridge is not None:
        accepted_tokens = [vocab_bridge.target_to_draft_id(t) for t in accepted_tokens]
        if bonus is not None:
            bonus = vocab_bridge.target_to_draft_id(bonus)

    logger.debug(
        "Verify RPC completed: round=%d, accepted=%d/%d, correction=%s, rpc_ms=%.2f",
        draft_tree.round_idx,
        response.num_accepted,
        len(draft_tree.token_ids),
        bonus,
        rpc_sw.elapsed_ms,
    )
    vr = VerificationResult(
        accepted_tokens=accepted_tokens,
        bonus_token=bonus,
        accepted_length=response.num_accepted,
        cache_hit=response.cache_hit,
        round_idx=draft_tree.round_idx,
    )
    return _RpcResult(value=vr, elapsed_ms=rpc_sw.elapsed_ms)


async def _call_flush_target_cache(
    target_stub: Any,
    session_id: str,
) -> None:
    """Send an ``EndSession`` signal to flush the Target Worker's KV cache.

    Called when speculation is invalidated and the speculative cache
    must be cleared for the session.

    Args:
        target_stub: A gRPC stub for TargetService (sync or async).
        session_id: Session ID to flush.
    """
    request = spec_decoding_pb2.EndSessionRequest(session_id=session_id)
    try:
        response = target_stub.EndSession(request)
        if inspect.isawaitable(response):
            response = await response
        logger.debug(
            "Cache flush completed: session=%s, was_active=%s",
            session_id,
            response.was_active,
        )
    except Exception:
        # EndSession is best-effort; don't break the pipeline
        logger.debug("Cache flush failed (non-critical): session=%s", session_id)


# ============================================================================
# Core: Overlapped Async Speculative Loop
# ============================================================================


async def run_speculative_loop_async(
    draft_stub: Any,
    target_stub: Any,
    prompt_ids: list[int],
    config: OrchestratorConfig | None = None,
    session_id: str = "default",
    eos_token_id: int | None = None,
    vocab_bridge: Any | None = None,
) -> PipelineResult:
    """Run the overlapped speculative decoding loop.

    This is the main entry point for the async pipeline.  It generates
    tokens by repeatedly:
        1. Firing ``VerifyDrafts(Tree N)`` as an async task.
        2. **Speculatively** firing ``GenerateDrafts(Tree N+1)`` assuming
           the longest branch of Tree N is accepted.
        3. Awaiting verification.
        4. If speculation was correct: use Tree N+1 directly.
        5. If speculation was wrong: discard Tree N+1, flush Draft cache,
           re-draft from corrected context.

    Args:
        draft_stub: gRPC stub for the Draft Service (sync or async).
        target_stub: gRPC stub for the Target Service (sync or async).
        prompt_ids: Tokenized input prompt.
        config: Pipeline configuration (defaults, timeouts, limits).
        session_id: Session ID for KV cache reuse on the Target Worker.
        eos_token_id: Token ID that signals end-of-generation.
        vocab_bridge: Optional VocabBridge mapping token IDs if draft/target differ.

    Returns:
        A :class:`PipelineResult` with the generated tokens, statistics,
        and timing information.

    Example::

        import asyncio
        result = asyncio.run(run_speculative_loop_async(
            draft_stub=draft_channel,
            target_stub=target_channel,
            prompt_ids=[101, 2003, 1037],
        ))
        print(f"Generated {len(result.output_tokens)} tokens "
              f"in {result.total_rounds} rounds "
              f"({result.acceptance_rate:.1%} accepted)")
    """
    cfg = config or OrchestratorConfig()
    telemetry = TelemetryLogger(service_name="orchestrator-pipeline")
    pipeline_sw = Stopwatch()
    pipeline_sw.start()

    state = SpeculativeState(prompt_ids=list(prompt_ids))

    # ------------------------------------------------------------------
    # Phase 0: Generate the first draft tree (no overlap possible here)
    # ------------------------------------------------------------------
    logger.info(
        "Starting speculative loop: prompt_len=%d, max_rounds=%d, max_output=%d, k=%d, draft_temp=%.2f",
        len(prompt_ids),
        cfg.max_rounds,
        cfg.max_output_tokens,
        cfg.max_draft_tokens,
        cfg.draft_temperature,
    )

    state.current_k = cfg.max_draft_tokens

    current_context = list(prompt_ids)
    current_draft: DraftTree | None = None
    speculative_draft: DraftTree | None = None
    speculative_assumption: list[int] = []  # What we assumed N would accept

    # Issue 11: Track the verified context length for delta-only transmission
    verified_context_len: int = 0

    with telemetry.span("initial_draft", round_idx=0):
        try:
            initial_rpc = await _call_generate_drafts(
                draft_stub,
                prompt_ids,
                current_context,
                cfg,
                round_idx=0,
                draft_len=state.current_k,
                session_id=session_id,
            )
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.warning("Initial draft RPC timed out (timeout=%.1fs)", cfg.timeout_s)
                pipeline_sw.stop()
                return PipelineResult(
                    output_tokens=[], total_rounds=0, acceptance_rate=0.0,
                    speculation_hit_rate=0.0, wall_time_ms=pipeline_sw.elapsed_ms,
                )
            raise
        current_draft = initial_rpc.value
        state.total_rpc_time_ms += initial_rpc.elapsed_ms

    # ------------------------------------------------------------------
    # Main Loop: Overlapped Draft(N+1) ‖ Verify(N)
    # ------------------------------------------------------------------
    for round_idx in range(cfg.max_rounds):
        if state.is_finished:
            break

        if len(state.generated_tokens) >= cfg.max_output_tokens:
            logger.info(
                "Reached max output tokens (%d), stopping",
                cfg.max_output_tokens,
            )
            state.is_finished = True
            break

        # Issue 8: Context window guardrail — prevent CUDA/HF crash
        if len(current_context) + cfg.max_draft_tokens >= cfg.max_context_window:
            logger.warning(
                "Context window limit reached: current=%d + draft=%d >= max=%d. "
                "Terminating generation to prevent model crash.",
                len(current_context),
                cfg.max_draft_tokens,
                cfg.max_context_window,
            )
            state.is_finished = True
            break

        if current_draft is None:
            logger.warning("No draft tree available at round %d, stopping", round_idx)
            break

        # --------------------------------------------------------------
        # Step A: Launch Verify(N) and speculative Draft(N+1) concurrently
        # --------------------------------------------------------------
        # Extract the longest path to form our speculation assumption,
        # avoiding sending a flat multi-branch tree to the draft context.
        assumed_path = _get_longest_path(current_draft)
        # Do not speculate beyond EOS — truncate path and skip draft RPC if it ends at EOS.
        if eos_token_id is not None and eos_token_id in assumed_path:
            eos_idx = assumed_path.index(eos_token_id)
            assumed_path = assumed_path[: eos_idx + 1]
        speculative_context = current_context + assumed_path
        speculative_assumption = assumed_path

        # Issue 11: Compute delta tokens for bandwidth optimization
        if verified_context_len > 0 and session_id:
            delta_token_ids = current_context[verified_context_len:]
        else:
            delta_token_ids = None

        # Create concurrent tasks
        # Bug 3 fix: When delta_token_ids is populated, send an empty
        # prompt_token_ids to avoid redundantly transmitting the full
        # context. The target service's session cache already has the
        # prefix and only needs the new tokens.
        verify_context = [] if delta_token_ids is not None else current_context
        verify_task = asyncio.create_task(
            _call_verify_drafts(
                target_stub,
                verify_context,
                current_draft,
                session_id,
                cfg,
                new_token_ids=delta_token_ids,
                expected_prefix_length=verified_context_len,
                vocab_bridge=vocab_bridge,
            )
        )

        # Speculatively draft N+1 while N is being verified.
        # Skip draft RPC if path ends at EOS (nothing useful to speculate).
        if eos_token_id is not None and assumed_path and assumed_path[-1] == eos_token_id:
            async def _noop_draft() -> _RpcResult[DraftTree]:
                return _RpcResult(
                    value=DraftTree(
                        token_ids=[], topology_map=[], proto_nodes=[], round_idx=round_idx + 1
                    ),
                    elapsed_ms=0.0,
                )

            speculative_draft_task = asyncio.create_task(_noop_draft())
        else:
            speculative_draft_task = asyncio.create_task(
                _call_generate_drafts(
                    draft_stub,
                    prompt_ids,
                    speculative_context,
                    cfg,
                    round_idx=round_idx + 1,
                    draft_len=state.current_k,
                    session_id=session_id,
                )
            )

        # Wait for BOTH to complete (they run concurrently)
        # Issue 9: Catch DEADLINE_EXCEEDED from either RPC
        with telemetry.span(
            "overlapped_round",
            round_idx=round_idx,
            draft_tokens=len(current_draft.token_ids),
        ) as ctx:
            try:
                verify_rpc_result, speculative_draft_rpc_result = await asyncio.gather(
                    verify_task,
                    speculative_draft_task,
                )
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.warning(
                        "RPC timed out at round %d (timeout=%.1fs). "
                        "Terminating pipeline gracefully.",
                        round_idx,
                        cfg.timeout_s,
                    )
                    state.is_finished = True
                    # Cancel both tasks to avoid "Task was destroyed but it is pending"
                    for t in (verify_task, speculative_draft_task):
                        t.cancel()
                    for t in (verify_task, speculative_draft_task):
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
                    break

                # Cache-eviction retry: target rejected delta-only payload
                # because the session was evicted (TTL/LRU). Retry with
                # full context so the target can rebuild its cache.
                # PR-4 Fix: The python TargetEngine raises CacheDesyncError,
                # which gRPC translates to StatusCode.UNKNOWN. We must check
                # both that or FAILED_PRECONDITION depending on interceptors.
                if (
                    e.code() in (grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.UNKNOWN)
                    and "CACHE_EVICTED_DELTA_ONLY" in (e.details() or "")
                ):
                    logger.warning(
                        "Target cache evicted for session %s at round %d. "
                        "Retrying verify with full context.",
                        session_id,
                        round_idx,
                    )
                    # Cancel the speculative draft — we'll re-draft after retry
                    speculative_draft_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await speculative_draft_task

                    # Retry with full context (no deltas)
                    retry_result = await _call_verify_drafts(
                        target_stub,
                        current_context,
                        current_draft,
                        session_id,
                        cfg,
                        new_token_ids=None,
                        expected_prefix_length=0,
                        vocab_bridge=vocab_bridge,
                    )
                    verify_rpc_result = retry_result
                    # Only add the retry time to the total sum
                    state.total_rpc_time_ms += retry_result.elapsed_ms

                    # Re-issue speculative draft
                    speculative_draft_rpc_result = await _call_generate_drafts(
                        draft_stub,
                        prompt_ids,
                        speculative_context,
                        cfg,
                        round_idx=round_idx + 1,
                        draft_len=state.current_k,
                        session_id=session_id,
                    )
                    state.total_rpc_time_ms += speculative_draft_rpc_result.elapsed_ms
                else:
                    # Cancel both tasks before re-raising to avoid task leak
                    for t in (verify_task, speculative_draft_task):
                        t.cancel()
                    for t in (verify_task, speculative_draft_task):
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
                    raise

            # Unwrap _RpcResult and accumulate RPC times
            verify_result = verify_rpc_result.value
            speculative_draft = speculative_draft_rpc_result.value

            # Issue 16: For overlapped RPCs, the wall-clock wait time is the max, not the sum.
            # elapsed_ms already includes simulated_rtt (sleep is inside RPC stopwatch).
            actual_wait_ms = max(
                verify_rpc_result.elapsed_ms,
                speculative_draft_rpc_result.elapsed_ms
            )
            state.total_rpc_time_ms += actual_wait_ms

            # Log extended structured telemetry metrics (PR-11)
            ctx.metadata["verify_rpc_ms"] = verify_rpc_result.elapsed_ms
            ctx.metadata["draft_rpc_ms"] = speculative_draft_rpc_result.elapsed_ms
            ctx.metadata["overlap_savings_ms"] = (
                verify_rpc_result.elapsed_ms + speculative_draft_rpc_result.elapsed_ms - actual_wait_ms
            )
            # Simple approximation of payload serialization size
            ctx.metadata["payload_bytes"] = len(current_draft.token_ids) * 8 + len(current_context) * 4
            ctx.metadata["rtt_ms_measured"] = actual_wait_ms
            ctx.metadata["k_pid_error"] = state.pid.prev_error
            ctx.metadata["k_pid_integral"] = state.pid.integral
            ctx.metadata["current_k"] = state.current_k

        # --------------------------------------------------------------
        # Step B: Process verification result
        # --------------------------------------------------------------
        state.total_rounds += 1
        state.total_tree_nodes += len(current_draft.token_ids)
        # Use the longest-path depth (not total tree nodes) for acceptance rate
        path_depth = len(assumed_path)
        state.total_path_depth += path_depth
        state.total_accepted += verify_result.accepted_length

        # Issue 14: Record per-round acceptance metrics
        round_acc = (
            verify_result.accepted_length / path_depth if path_depth > 0 else 0.0
        )
        state.per_round_metrics.append(
            RoundMetrics(
                round_idx=round_idx,
                accepted=verify_result.accepted_length,
                path_depth=path_depth,
                tree_nodes=len(current_draft.token_ids),
                acceptance_rate=round_acc,
            )
        )
        
        # Update rolling acceptance rate queue
        state.recent_acceptance_rates.append(round_acc)
        rolling_mean = sum(state.recent_acceptance_rates) / len(state.recent_acceptance_rates)

        # Calculate dynamic fallback threshold (roughly alpha * 0.8)
        # alpha is breakeven rate: proxy with actual measured RTT
        measured_rtt = actual_wait_ms
        t_target_approx = 15.0 # Rough approximation of target forward pass
        breakeven_alpha = measured_rtt / t_target_approx if measured_rtt > 0 else 0.5
        fallback_threshold = min(0.5, breakeven_alpha * 0.8)

        # PR-7 / PR-8: Dynamic K Adjustment and Fallback Mode
        # Only adjust after we have at least 3 samples
        if len(state.recent_acceptance_rates) >= 3:
            # PR-8: Fallback Circuit Breaker
            if rolling_mean < fallback_threshold:
                state.consecutive_low_acceptance += 1
                if state.consecutive_low_acceptance >= 2:
                    if not state.is_fallback_mode:
                        logger.warning("Acceptance rate collapsed (%.2f < %.2f). Entering Fallback Mode.", rolling_mean, fallback_threshold)
                    state.is_fallback_mode = True
            else:
                state.consecutive_low_acceptance = 0
                if state.is_fallback_mode and rolling_mean > fallback_threshold + 0.1:
                    logger.info("Acceptance rate recovered (%.2f). Exiting Fallback Mode.", rolling_mean)
                    state.is_fallback_mode = False

            # PR-7: Adaptive K using PID
            if not state.is_fallback_mode:
                error = rolling_mean - state.pid.setpoint
                state.pid.integral += error
                derivative = error - state.pid.prev_error
                
                adjustment = (
                    state.pid.kp * error +
                    state.pid.ki * state.pid.integral +
                    state.pid.kd * derivative
                )
                state.pid.prev_error = error
                
                # Apply adjustment
                new_k = int(round(state.current_k + adjustment))
                new_k = max(1, min(new_k, cfg.max_draft_tokens))
                
                if new_k != state.current_k:
                    logger.debug("PID adjusted K: %d -> %d (err=%.2f, adj=%.2f)", state.current_k, new_k, error, adjustment)
                    state.current_k = new_k

        # Append accepted tokens and bonus token to output
        new_tokens = list(verify_result.accepted_tokens)
        # Safely check for None as 0 is a valid token ID
        if verify_result.bonus_token is not None:
            new_tokens.append(verify_result.bonus_token)

        # Check for EOS in newly generated tokens
        if eos_token_id is not None and eos_token_id in new_tokens:
            logger.info("EOS token found at round %d", round_idx)
            # Truncate at EOS
            eos_pos = new_tokens.index(eos_token_id)
            new_tokens = new_tokens[: eos_pos + 1]
            state.is_finished = True

        state.generated_tokens.extend(new_tokens)
        # Context for next round is only verified output (never speculative draft).
        current_context = list(prompt_ids) + state.generated_tokens
        
        # Issue 11 & PR-2 Fix: Update verified_context_len for delta-only transmission.
        # The target's seq_len ONLY includes the accepted tokens, NOT the bonus token
        # (which is sent to the target as a delta in the NEXT round). So we must align 
        # expected_prefix_length to maintain perfect sync with the Target KV Cache.
        bonus_count = 1 if verify_result.bonus_token is not None else 0
        verified_context_len = len(current_context) - bonus_count

        logger.info(
            "Round %d: accepted %d/%d (path_depth=%d, tree_nodes=%d), bonus=%s, total_output=%d",
            round_idx,
            verify_result.accepted_length,
            path_depth,
            path_depth,
            len(current_draft.token_ids),
            verify_result.bonus_token,
            len(state.generated_tokens),
        )

        if state.is_finished:
            break

        # --------------------------------------------------------------
        # Step C: Check if our speculation was correct
        # --------------------------------------------------------------
        # The speculative draft N+1 was computed from context:
        #   speculative_context = current_context + assumed_path
        # Its first root token predicts position len(speculative_context).
        #
        # After verification the actual context becomes:
        #   current_context + accepted_tokens + bonus_token
        # The bonus token occupies the SAME position that the speculative
        # draft's first root predicts.
        #
        # For the speculative draft to be valid, BOTH must hold:
        #   1. accepted_tokens == assumed_path  (prefix is correct)
        #   2. bonus_token == speculative_draft.root  (continuation matches)
        # If (1) holds but (2) doesn't, the draft tree was generated from
        # a context that diverges at the very first position — the target
        # would reject everything, wasting a verify call.
        actual_accepted = list(verify_result.accepted_tokens)
        # NOTE: This comparison uses token IDs only. In theory, two
        # different branches in the tree could have identical token ID
        # sequences, producing a false-positive speculation hit. This
        # is negligible for real LLMs (vocab 32K+) but could matter
        # for small-vocabulary CI models with high branching factor.
        path_matches = actual_accepted == list(speculative_assumption)

        bonus_matches = False
        matched_root_idx = None
        if path_matches and speculative_draft is not None and speculative_draft.token_ids:
            # Fix B1: Check ALL root nodes, not just the first one.
            # The target model might have picked any of the root branches.
            root_indices = [
                i for i, p in enumerate(speculative_draft.topology_map) if p == -1
            ]
            for root_idx in root_indices:
                if verify_result.bonus_token == speculative_draft.token_ids[root_idx]:
                    bonus_matches = True
                    matched_root_idx = root_idx
                    break

        speculation_correct = path_matches and bonus_matches

        if speculation_correct:
            # 🎉 Speculation hit!  Use the pre-computed draft N+1.
            state.speculation_hits += 1

            # Prune the draft tree for ALL matched roots (including primary idx 0).
            # Promote the matched root's children to new roots and discard the
            # matched node. The bonus token is already appended; the next round
            # must compare target's prediction *after* bonus against draft's
            # children (guesses for next token), not the bonus itself.
            if matched_root_idx is not None:
                old_ids = speculative_draft.token_ids
                old_topo = speculative_draft.topology_map
                # BFS from CHILDREN of matched root (promote them to roots)
                children_of_matched = [
                    j for j, p in enumerate(old_topo) if p == matched_root_idx
                ]
                if not children_of_matched:
                    # Matched root is a leaf — no subtree to reuse; re-draft
                    current_draft = None
                    state.speculation_hits -= 1
                    state.speculation_misses += 1
                    logger.info(
                        "Speculation HIT at round %d — matched root is leaf, "
                        "re-drafting from corrected context",
                        round_idx,
                    )
                    # Target has already rolled back to accepted prefix; no flush needed.
                    with telemetry.span("re_draft", round_idx=round_idx):
                        try:
                            re_draft_rpc = await _call_generate_drafts(
                                draft_stub,
                                prompt_ids,
                                current_context,
                                cfg,
                                round_idx=round_idx + 1,
                                session_id=session_id,
                            )
                        except grpc.aio.AioRpcError as e:
                            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                                logger.warning(
                                    "Re-draft RPC timed out at round %d.",
                                    round_idx,
                                )
                                state.is_finished = True
                                break
                            raise
                        current_draft = re_draft_rpc.value
                        state.total_rpc_time_ms += re_draft_rpc.elapsed_ms
                    speculative_draft = None
                else:
                    keep: set[int] = set()
                    
                    children_map: dict[int, list[int]] = {}
                    for i, parent_idx in enumerate(old_topo):
                        children_map.setdefault(parent_idx, []).append(i)

                    queue_prune = list(children_of_matched)
                    while queue_prune:
                        node = queue_prune.pop(0)
                        keep.add(node)
                        for child in children_map.get(node, []):
                            if child not in keep:
                                queue_prune.append(child)
                    sorted_keep = sorted(keep)
                    old_to_new = {old: new for new, old in enumerate(sorted_keep)}
                    new_ids = [old_ids[i] for i in sorted_keep]
                    new_topo = [
                        -1 if old_topo[i] == matched_root_idx
                        else old_to_new.get(old_topo[i], -1)
                        for i in sorted_keep
                    ]
                    # Preserve top_k and log_probs from original tree for correct
                    # residual sampling and stochastic acceptance (q_val = exp(log_prob)).
                    _, _, orig_top_k, orig_log_probs = _flatten_proto_tree_with_topk(
                        speculative_draft.proto_nodes
                    )
                    new_top_k = [orig_top_k[old_idx] for old_idx in sorted_keep]
                    new_log_probs = [orig_log_probs[old_idx] for old_idx in sorted_keep]
                    proto_nodes = _arrays_to_proto_nodes(
                        new_ids, new_topo, top_k_per_node=new_top_k, log_probs=new_log_probs
                    )
                    current_draft = DraftTree(
                        token_ids=new_ids,
                        topology_map=new_topo,
                        proto_nodes=proto_nodes,
                        round_idx=speculative_draft.round_idx,
                    )
                    logger.info(
                        "Speculation HIT at round %d — reusing draft N+1 "
                        "(pruned to children of root %d, %d→%d nodes)",
                        round_idx,
                        matched_root_idx,
                        len(old_ids),
                        len(new_ids),
                    )
            else:
                current_draft = speculative_draft
                logger.info(
                    "Speculation HIT at round %d — reusing draft N+1",
                    round_idx,
                )
        else:
            # ❌ Speculation miss.  The speculative draft was computed
            # from an incorrect context.  Discard it and re-draft.
            state.speculation_misses += 1
            logger.info(
                "Speculation MISS at round %d — discarding draft N+1 "
                "(accepted %d/%d assumed tokens)",
                round_idx,
                len(actual_accepted),
                len(speculative_assumption),
            )

            # Target has already rolled back to accepted prefix. Draft engine uses
            # LCP cache slicing for O(1) rollback — no reset_cache needed.

            # Re-draft from the CORRECTED context
            with telemetry.span("re_draft", round_idx=round_idx):
                try:
                    re_draft_rpc = await _call_generate_drafts(
                        draft_stub,
                        prompt_ids,
                        current_context,
                        cfg,
                        round_idx=round_idx + 1,
                        draft_len=state.current_k,
                        session_id=session_id,
                    )
                except grpc.aio.AioRpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        logger.warning(
                            "Re-draft RPC timed out at round %d. Terminating.",
                            round_idx,
                        )
                        state.is_finished = True
                        break
                    raise
                current_draft = re_draft_rpc.value
                state.total_rpc_time_ms += re_draft_rpc.elapsed_ms
            speculative_draft = None

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    pipeline_sw.stop()

    total_spec = state.speculation_hits + state.speculation_misses
    spec_hit_rate = state.speculation_hits / total_spec if total_spec > 0 else 0.0
    acceptance_rate = (
        state.total_accepted / state.total_path_depth
        if state.total_path_depth > 0
        else 0.0
    )

    result = PipelineResult(
        output_tokens=state.generated_tokens,
        total_rounds=state.total_rounds,
        acceptance_rate=acceptance_rate,
        speculation_hit_rate=spec_hit_rate,
        wall_time_ms=pipeline_sw.elapsed_ms,
        network_idle_ms=state.total_rpc_time_ms,
        per_round_acceptance=[
            {
                "round": m.round_idx,
                "accepted": m.accepted,
                "path_depth": m.path_depth,
                "tree_nodes": m.tree_nodes,
                "acceptance_rate": round(m.acceptance_rate, 4),
            }
            for m in state.per_round_metrics
        ],
        telemetry=[s.to_dict() for s in telemetry.spans],
    )

    logger.info(
        "Pipeline complete: %d tokens in %d rounds, "
        "acceptance=%.1f%%, speculation_hits=%.1f%%, "
        "wall_time=%.1f ms",
        len(result.output_tokens),
        result.total_rounds,
        result.acceptance_rate * 100,
        result.speculation_hit_rate * 100,
        result.wall_time_ms,
    )

    return result
