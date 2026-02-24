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
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from specsplit.core.config import OrchestratorConfig
from specsplit.core.telemetry import Stopwatch, TelemetryLogger
from specsplit.proto import spec_decoding_pb2

logger = logging.getLogger(__name__)


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
    bonus_token: int
    accepted_length: int
    cache_hit: bool = False
    round_idx: int = 0


@dataclass
class SpeculativeState:
    """Tracks the current state of the speculative pipeline.

    Attributes:
        generated_tokens: All tokens generated so far (accepted + bonus).
        prompt_ids: The original prompt token IDs.
        total_rounds: Number of verify rounds completed.
        total_accepted: Total draft tokens accepted across all rounds.
        total_drafted: Total draft tokens generated across all rounds.
        speculation_hits: Number of times N+1 speculation was correct.
        speculation_misses: Number of times N+1 was discarded.
        is_finished: Whether generation has reached a stop condition.
    """

    generated_tokens: list[int] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    total_rounds: int = 0
    total_accepted: int = 0
    total_drafted: int = 0
    speculation_hits: int = 0
    speculation_misses: int = 0
    is_finished: bool = False


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

    def _walk(node: Any, parent_idx: int) -> None:
        my_idx = len(token_ids)
        token_ids.append(node.token_id)
        topology_map.append(parent_idx)
        for child in node.children:
            _walk(child, my_idx)

    for root in proto_nodes:
        _walk(root, -1)

    return token_ids, topology_map


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
) -> DraftTree:
    """Call ``DraftService.GenerateDrafts`` over gRPC.

    Constructs a ``DraftRequest`` protobuf, sends it to the Draft Worker,
    and converts the ``DraftResponse`` into a :class:`DraftTree`.

    Args:
        draft_stub: A gRPC stub for DraftService (sync or async).
        prompt_ids: Original prompt token IDs.
        context_ids: Current context (prompt + generated so far).
        config: Pipeline configuration.
        round_idx: Current round index (for tracing).

    Returns:
        A :class:`DraftTree` produced by the Draft Worker.
    """
    # Inject Synthetic RTT Latency (Task 4.1)
    if config.simulated_rtt_ms > 0:
        await asyncio.sleep(config.simulated_rtt_ms / 1000.0)

    request = spec_decoding_pb2.DraftRequest(
        request_id=f"round-{round_idx}-{uuid.uuid4().hex[:8]}",
        prompt_token_ids=context_ids,
        max_draft_len=config.max_draft_tokens,
    )

    response = draft_stub.GenerateDrafts(request)
    # Support both sync and async stubs
    if asyncio.iscoroutine(response) or asyncio.isfuture(response):
        response = await response

    # Convert the nested proto tree to flat arrays
    proto_nodes = list(response.draft_tree)
    token_ids, topology_map = _flatten_proto_tree(proto_nodes)

    logger.debug(
        "Draft RPC completed: round=%d, tokens=%d",
        round_idx,
        len(token_ids),
    )
    return DraftTree(
        token_ids=token_ids,
        topology_map=topology_map,
        proto_nodes=proto_nodes,
        round_idx=round_idx,
    )


async def _call_verify_drafts(
    target_stub: Any,
    context_ids: list[int],
    draft_tree: DraftTree,
    session_id: str,
    config: OrchestratorConfig,
) -> VerificationResult:
    """Call ``TargetService.VerifyDrafts`` over gRPC.

    Forwards the draft tree's proto nodes directly to the Target Worker
    to avoid re-encoding.

    Args:
        target_stub: A gRPC stub for TargetService (sync or async).
        context_ids: Full current context (prompt + generated so far).
        draft_tree: The draft tree to verify (with preserved proto nodes).
        session_id: Session ID for KV cache reuse.
        config: Pipeline configuration.

    Returns:
        A :class:`VerificationResult` from the Target Worker.
    """
    # Inject Synthetic RTT Latency (Task 4.1)
    if config.simulated_rtt_ms > 0:
        await asyncio.sleep(config.simulated_rtt_ms / 1000.0)
        
    request = spec_decoding_pb2.VerifyRequest(
        request_id=f"verify-{draft_tree.round_idx}-{uuid.uuid4().hex[:8]}",
        prompt_token_ids=context_ids,
        draft_tree=draft_tree.proto_nodes,
        session_id=session_id,
    )

    response = target_stub.VerifyDrafts(request)
    # Support both sync and async stubs
    if asyncio.iscoroutine(response) or asyncio.isfuture(response):
        response = await response

    accepted_tokens = list(response.accepted_token_ids)
    correction = response.correction_token_id if response.correction_token_id != 0 else None

    logger.debug(
        "Verify RPC completed: round=%d, accepted=%d/%d, correction=%s",
        draft_tree.round_idx,
        response.num_accepted,
        len(draft_tree.token_ids),
        correction,
    )
    return VerificationResult(
        accepted_tokens=accepted_tokens,
        bonus_token=response.correction_token_id,
        accepted_length=response.num_accepted,
        cache_hit=response.cache_hit,
        round_idx=draft_tree.round_idx,
    )


async def _call_flush_draft_cache(
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
        if asyncio.iscoroutine(response) or asyncio.isfuture(response):
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
        "Starting speculative loop: prompt_len=%d, max_rounds=%d, max_output=%d",
        len(prompt_ids),
        cfg.max_rounds,
        cfg.max_output_tokens,
    )

    current_context = list(prompt_ids)
    current_draft: DraftTree | None = None
    speculative_draft: DraftTree | None = None
    speculative_assumption: list[int] = []  # What we assumed N would accept

    with telemetry.span("initial_draft", round_idx=0):
        current_draft = await _call_generate_drafts(
            draft_stub,
            prompt_ids,
            current_context,
            cfg,
            round_idx=0,
        )

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

        if current_draft is None:
            logger.warning("No draft tree available at round %d, stopping", round_idx)
            break

        # --------------------------------------------------------------
        # Step A: Launch Verify(N) and speculative Draft(N+1) concurrently
        # --------------------------------------------------------------
        # Extract the longest path to form our speculation assumption, 
        # avoiding sending a flat multi-branch tree to the draft context.
        assumed_path = _get_longest_path(current_draft)
        speculative_context = current_context + assumed_path
        speculative_assumption = assumed_path

        # Create concurrent tasks
        verify_task = asyncio.create_task(
            _call_verify_drafts(
                target_stub,
                current_context,
                current_draft,
                session_id,
                cfg,
            )
        )

        # Speculatively draft N+1 while N is being verified
        speculative_draft_task = asyncio.create_task(
            _call_generate_drafts(
                draft_stub,
                prompt_ids,
                speculative_context,
                cfg,
                round_idx=round_idx + 1,
            )
        )

        # Wait for BOTH to complete (they run concurrently)
        with telemetry.span(
            "overlapped_round",
            round_idx=round_idx,
            draft_tokens=len(current_draft.token_ids),
        ):
            verify_result, speculative_draft = await asyncio.gather(
                verify_task,
                speculative_draft_task,
            )

        # --------------------------------------------------------------
        # Step B: Process verification result
        # --------------------------------------------------------------
        state.total_rounds += 1
        state.total_drafted += len(current_draft.token_ids)
        state.total_accepted += verify_result.accepted_length

        # Append accepted tokens and bonus token to output
        new_tokens = list(verify_result.accepted_tokens)
        if verify_result.bonus_token is not None and verify_result.bonus_token != 0:
            new_tokens.append(verify_result.bonus_token)

        # Check for EOS in newly generated tokens
        if eos_token_id is not None and eos_token_id in new_tokens:
            logger.info("EOS token found at round %d", round_idx)
            # Truncate at EOS
            try:
                eos_pos = new_tokens.index(eos_token_id)
                new_tokens = new_tokens[: eos_pos + 1]
            except ValueError:
                pass
            state.is_finished = True

        state.generated_tokens.extend(new_tokens)
        # Context for next round is only verified output (never speculative draft).
        current_context = list(prompt_ids) + state.generated_tokens

        logger.debug(
            "Round %d: accepted %d/%d, bonus=%d, total_output=%d",
            round_idx,
            verify_result.accepted_length,
            len(current_draft.token_ids),
            verify_result.bonus_token,
            len(state.generated_tokens),
        )

        if state.is_finished:
            break

        # --------------------------------------------------------------
        # Step C: Check if our speculation was correct
        # --------------------------------------------------------------
        actual_accepted = list(verify_result.accepted_tokens)
        speculation_correct = actual_accepted == speculative_assumption[
            : len(actual_accepted)
        ]

        if speculation_correct:
            # 🎉 Speculation hit!  Use the pre-computed draft N+1.
            state.speculation_hits += 1
            current_draft = speculative_draft
            logger.debug("Speculation HIT at round %d — reusing draft N+1", round_idx)
        else:
            # ❌ Speculation miss.  The speculative draft was computed
            # from an incorrect context.  Discard it and re-draft.
            state.speculation_misses += 1
            logger.debug(
                "Speculation MISS at round %d — discarding draft N+1 (accepted %d/%d assumed)",
                round_idx,
                len(actual_accepted),
                len(speculative_assumption),
            )

            # Send flush signal to Target Worker to clear its speculative cache (no-op when stateless)
            if session_id:
                await _call_flush_draft_cache(target_stub, session_id)

            # Re-draft from the CORRECTED context
            with telemetry.span("re_draft", round_idx=round_idx):
                current_draft = await _call_generate_drafts(
                    draft_stub,
                    prompt_ids,
                    current_context,
                    cfg,
                    round_idx=round_idx + 1,
                )
            speculative_draft = None

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    pipeline_sw.stop()

    total_spec = state.speculation_hits + state.speculation_misses
    spec_hit_rate = state.speculation_hits / total_spec if total_spec > 0 else 0.0
    acceptance_rate = state.total_accepted / state.total_drafted if state.total_drafted > 0 else 0.0

    result = PipelineResult(
        output_tokens=state.generated_tokens,
        total_rounds=state.total_rounds,
        acceptance_rate=acceptance_rate,
        speculation_hit_rate=spec_hit_rate,
        wall_time_ms=pipeline_sw.elapsed_ms,
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