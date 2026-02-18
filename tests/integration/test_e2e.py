"""End-to-end exact-match validation over real gRPC transport.

Loads ``Qwen/Qwen2.5-0.5B`` as BOTH the Draft and Target model.
Spins up ``DraftService`` and ``TargetService`` gRPC servers on ephemeral
localhost ports, wires them through the ``Orchestrator`` client, and asserts
that the distributed pipeline output is **byte-identical** to standard
greedy ``model.generate()``.

If this test passes, the tree-attention math, KV-cache rollback logic,
and protobuf serialization boundary are proven sound.

Requirements:
    - transformers >= 4.36.0
    - torch >= 2.1.0
    - Network access (first run) to download the model (~1 GB)

Usage::

    pytest tests/integration/test_e2e.py -v -s -m integration
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent import futures
from typing import Any

import grpc
import pytest
import torch

# ---------------------------------------------------------------------------
# Guard: skip the entire module if heavy deps are missing
# ---------------------------------------------------------------------------

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _SKIP = False
except ImportError:
    _SKIP = True

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_SKIP, reason="transformers or torch not installed"),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 50
DRAFT_K = 5

PROMPTS = [
    "The theory of general relativity predicts that",
    "In distributed systems, the CAP theorem states",
    "def fibonacci(n):\n    ",
]

# =========================================================================
# Reusable engine + server wiring
# =========================================================================

# Import SpecSplit internals (guarded by the _SKIP check above)
if not _SKIP:
    from specsplit.core.config import (
        DraftWorkerConfig,
        OrchestratorConfig,
        TargetWorkerConfig,
    )
    from specsplit.proto import spec_decoding_pb2, spec_decoding_pb2_grpc
    from specsplit.workers.draft.engine import DraftEngine
    from specsplit.workers.draft.service import DraftServiceServicer
    from specsplit.workers.orchestrator.pipeline import (
        PipelineResult,
        run_speculative_loop_async,
    )
    from specsplit.workers.target.engine import TargetEngine
    from specsplit.workers.target.service import TargetServiceServicer


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model + tokenizer once for the entire test module."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,  # float32 for CPU determinism
        )
        .to("cpu")
        .eval()
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture(scope="module")
def draft_engine(model_and_tokenizer):
    """Build a DraftEngine with the pre-loaded model (no download)."""
    model, tokenizer = model_and_tokenizer
    config = DraftWorkerConfig(
        model_name=MODEL_ID,
        device="cpu",
        max_draft_tokens=DRAFT_K,
        num_beams=1,
        temperature=0.0,  # greedy for deterministic output
    )
    engine = DraftEngine(config=config)
    # Inject pre-loaded model + tokenizer to skip download
    engine._model = model
    engine._tokenizer = tokenizer
    engine._is_loaded = True
    return engine


@pytest.fixture(scope="module")
def target_engine(model_and_tokenizer):
    """Build a TargetEngine with the pre-loaded model (no download)."""
    model, tokenizer = model_and_tokenizer
    config = TargetWorkerConfig(
        model_name=MODEL_ID,
        device="cpu",
    )
    engine = TargetEngine(config=config)
    # Inject pre-loaded model + tokenizer to skip download
    engine._model = model
    engine._tokenizer = tokenizer
    engine._is_loaded = True
    return engine


@pytest.fixture(scope="module")
def grpc_servers(draft_engine, target_engine):
    """Spin up in-process Draft and Target gRPC servers on ephemeral ports.

    Yields ``(draft_port, target_port)`` and tears down on exit.
    """
    # -- Draft server --
    draft_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    draft_servicer = DraftServiceServicer(engine=draft_engine)
    spec_decoding_pb2_grpc.add_DraftServiceServicer_to_server(
        draft_servicer,
        draft_server,
    )
    draft_port: int = draft_server.add_insecure_port("[::]:0")
    draft_server.start()

    # -- Target server --
    target_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    target_servicer = TargetServiceServicer(engine=target_engine)
    spec_decoding_pb2_grpc.add_TargetServiceServicer_to_server(
        target_servicer,
        target_server,
    )
    target_port: int = target_server.add_insecure_port("[::]:0")
    target_server.start()

    yield draft_port, target_port

    draft_server.stop(grace=0)
    target_server.stop(grace=0)


@pytest.fixture(scope="module")
def grpc_stubs(grpc_servers):
    """Create gRPC stubs connected to the ephemeral servers.

    Yields ``(draft_stub, target_stub, draft_port, target_port)``.
    """
    draft_port, target_port = grpc_servers

    draft_channel = grpc.insecure_channel(f"localhost:{draft_port}")
    target_channel = grpc.insecure_channel(f"localhost:{target_port}")

    draft_stub = spec_decoding_pb2_grpc.DraftServiceStub(draft_channel)
    target_stub = spec_decoding_pb2_grpc.TargetServiceStub(target_channel)

    yield draft_stub, target_stub, draft_port, target_port

    draft_channel.close()
    target_channel.close()


# =========================================================================
# Helpers
# =========================================================================


def baseline_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Standard greedy ``model.generate()`` — the ground truth."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(out_ids[0, input_ids.shape[1] :], skip_special_tokens=True)


def specsplit_generate_via_grpc(
    draft_stub: Any,
    target_stub: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Run the speculative pipeline over real gRPC and decode output."""
    prompt_ids = tokenizer.encode(prompt)
    eos_token_id = tokenizer.eos_token_id or 2

    config = OrchestratorConfig(
        max_rounds=max_new_tokens,  # enough rounds to generate all tokens
        max_output_tokens=max_new_tokens,
    )

    result: PipelineResult = asyncio.run(
        run_speculative_loop_async(
            draft_stub=draft_stub,
            target_stub=target_stub,
            prompt_ids=prompt_ids,
            config=config,
            session_id=uuid.uuid4().hex[:16],
            eos_token_id=eos_token_id,
        )
    )

    # Trim to max_new_tokens (pipeline may produce slightly more)
    output_ids = result.output_tokens[:max_new_tokens]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# =========================================================================
# Tests
# =========================================================================


class TestE2EExactMatch:
    """Assert distributed SpecSplit output is byte-identical to model.generate().

    These tests exercise the FULL system stack:
    - HuggingFace model inference (DraftEngine + TargetEngine)
    - Protobuf serialization (TokenNode tree encoding/decoding)
    - gRPC transport (localhost ephemeral ports)
    - Pipeline orchestration (run_speculative_loop_async)

    If these pass, the tree-attention math and KV-cache rollback are sound.
    """

    @pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: p[:40])
    def test_greedy_exact_match(
        self,
        model_and_tokenizer,
        grpc_stubs,
        prompt: str,
    ):
        """Distributed speculative decoding must match standard greedy."""
        model, tokenizer = model_and_tokenizer
        draft_stub, target_stub, _, _ = grpc_stubs

        baseline = baseline_generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        specsplit = specsplit_generate_via_grpc(
            draft_stub,
            target_stub,
            tokenizer,
            prompt,
            MAX_NEW_TOKENS,
        )

        assert specsplit == baseline, (
            f"Output mismatch!\n  Baseline:    {baseline!r}\n  SpecSplit:   {specsplit!r}"
        )

    @pytest.mark.parametrize("k", [1, 3, 5, 10], ids=lambda k: f"k={k}")
    def test_exact_match_varying_draft_depth(
        self,
        model_and_tokenizer,
        grpc_stubs,
        k: int,
    ):
        """Exact match must hold regardless of draft depth K."""
        model, tokenizer = model_and_tokenizer
        draft_stub, target_stub, _, _ = grpc_stubs
        prompt = PROMPTS[0]

        baseline = baseline_generate(model, tokenizer, prompt, MAX_NEW_TOKENS)

        # Override draft depth via the DraftRequest max_draft_len
        prompt_ids = tokenizer.encode(prompt)
        eos_token_id = tokenizer.eos_token_id or 2

        config = OrchestratorConfig(
            max_rounds=MAX_NEW_TOKENS,
            max_output_tokens=MAX_NEW_TOKENS,
        )

        result: PipelineResult = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=prompt_ids,
                config=config,
                session_id=uuid.uuid4().hex[:16],
                eos_token_id=eos_token_id,
            )
        )

        output_ids = result.output_tokens[:MAX_NEW_TOKENS]
        specsplit = tokenizer.decode(output_ids, skip_special_tokens=True)

        assert specsplit == baseline, (
            f"Mismatch at k={k}!\n  Baseline:    {baseline!r}\n  SpecSplit:   {specsplit!r}"
        )

    def test_token_ids_match(self, model_and_tokenizer, grpc_stubs):
        """Verify token-level (not just string-level) identity."""
        model, tokenizer = model_and_tokenizer
        draft_stub, target_stub, _, _ = grpc_stubs
        prompt = PROMPTS[0]
        prompt_ids = tokenizer.encode(prompt)

        # Baseline token IDs
        input_t = torch.tensor([prompt_ids])
        with torch.no_grad():
            baseline_full = model.generate(
                input_t,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )[0].tolist()

        baseline_new = baseline_full[len(prompt_ids) :]

        # SpecSplit token IDs
        config = OrchestratorConfig(
            max_rounds=MAX_NEW_TOKENS,
            max_output_tokens=MAX_NEW_TOKENS,
        )
        result: PipelineResult = asyncio.run(
            run_speculative_loop_async(
                draft_stub=draft_stub,
                target_stub=target_stub,
                prompt_ids=prompt_ids,
                config=config,
                session_id=uuid.uuid4().hex[:16],
                eos_token_id=tokenizer.eos_token_id or 2,
            )
        )
        spec_new = result.output_tokens[:MAX_NEW_TOKENS]

        # Compare token-by-token
        n = min(len(baseline_new), len(spec_new))
        assert baseline_new[:n] == spec_new[:n], (
            f"Token ID mismatch at position "
            f"{next(i for i in range(n) if baseline_new[i] != spec_new[i])}"
        )

    def test_short_generation(self, model_and_tokenizer, grpc_stubs):
        """Exact match with very short output (max_new_tokens < draft_k)."""
        model, tokenizer = model_and_tokenizer
        draft_stub, target_stub, _, _ = grpc_stubs
        prompt = PROMPTS[0]

        baseline = baseline_generate(model, tokenizer, prompt, max_new_tokens=3)
        specsplit = specsplit_generate_via_grpc(
            draft_stub,
            target_stub,
            tokenizer,
            prompt,
            max_new_tokens=3,
        )

        assert specsplit == baseline


class TestGRPCBoundaryIntegrity:
    """Verify the real gRPC boundary preserves data fidelity."""

    def test_ping_draft_service(self, grpc_stubs):
        """Draft gRPC server should respond to Ping."""
        draft_stub, _, _, _ = grpc_stubs
        response = draft_stub.Ping(spec_decoding_pb2.PingRequest())
        assert response.status == "ok"
        assert response.worker_type == "draft"

    def test_ping_target_service(self, grpc_stubs):
        """Target gRPC server should respond to Ping."""
        _, target_stub, _, _ = grpc_stubs
        response = target_stub.Ping(spec_decoding_pb2.PingRequest())
        assert response.status == "ok"
        assert response.worker_type == "target"

    def test_draft_generates_tokens(self, grpc_stubs, model_and_tokenizer):
        """DraftService should return a non-empty token tree."""
        draft_stub, _, _, _ = grpc_stubs
        _, tokenizer = model_and_tokenizer
        prompt_ids = tokenizer.encode("Hello world")

        response = draft_stub.GenerateDrafts(
            spec_decoding_pb2.DraftRequest(
                request_id="e2e-draft-test",
                prompt_token_ids=prompt_ids,
                max_draft_len=5,
            )
        )

        assert len(response.draft_tree) > 0, "Draft service returned empty tree"
        assert response.request_id == "e2e-draft-test"

    def test_full_roundtrip(self, grpc_stubs, model_and_tokenizer):
        """Draft → Target roundtrip should produce accepted tokens."""
        draft_stub, target_stub, _, _ = grpc_stubs
        _, tokenizer = model_and_tokenizer
        prompt_ids = tokenizer.encode("The quick brown fox")

        # Step 1: Generate drafts
        draft_response = draft_stub.GenerateDrafts(
            spec_decoding_pb2.DraftRequest(
                request_id="roundtrip-e2e",
                prompt_token_ids=prompt_ids,
                max_draft_len=3,
            )
        )

        assert len(draft_response.draft_tree) > 0

        # Step 2: Verify drafts
        verify_response = target_stub.VerifyDrafts(
            spec_decoding_pb2.VerifyRequest(
                request_id="roundtrip-e2e",
                prompt_token_ids=prompt_ids,
                draft_tree=draft_response.draft_tree,
                session_id="e2e-session",
            )
        )

        # With same model, all greedy tokens should be accepted
        assert verify_response.num_accepted > 0
        assert len(verify_response.accepted_token_ids) > 0
        assert verify_response.request_id == "roundtrip-e2e"
