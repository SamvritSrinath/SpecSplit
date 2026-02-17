"""Exact-match validation: SpecSplit speculative decoding vs standard generation.

Loads a small model (Qwen/Qwen2.5-0.5B) as BOTH the Draft and Target model
locally. Verifies that the speculative decoding pipeline produces output
**strictly identical** to standard autoregressive ``model.generate()``.

The gRPC network boundary is mocked with in-process stubs so this can run
in CI without binding to actual ports.

Requirements:
    - transformers >= 4.36.0
    - torch >= 2.1.0
    - Network access (first run) to download the model (~1 GB)

Usage::

    pytest tests/integration/test_exact_match.py -v -s
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

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
DRAFT_K = 5  # speculative depth per round

PROMPTS = [
    "The theory of general relativity predicts that",
    "In distributed systems, the CAP theorem states",
    "def fibonacci(n):\n    ",
]

# =========================================================================
# Mock protobuf messages — mirrors spec_decoding.proto without codegen
# =========================================================================


@dataclass
class MockDraftRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_draft_len: int
    temperature: float = 0.0


@dataclass
class MockDraftResponse:
    request_id: str
    draft_token_ids: list[int]
    draft_log_probs: list[float]


@dataclass
class MockVerifyRequest:
    request_id: str
    prompt_token_ids: list[int]
    draft_token_ids: list[int]
    session_id: str = ""


@dataclass
class MockVerifyResponse:
    request_id: str
    accepted_token_ids: list[int]
    correction_token_id: int | None
    num_accepted: int
    cache_hit: bool = False


# =========================================================================
# Real inference engines (thin wrappers around HF model)
# =========================================================================


class LocalDraftEngine:
    """Greedy autoregressive draft generation using the real model."""

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate_drafts(self, prompt_ids: list[int], k: int) -> MockDraftResponse:
        """Generate *k* greedy draft tokens from *prompt_ids*."""
        input_ids = torch.tensor([prompt_ids], device=self.device)
        drafted: list[int] = []
        log_probs: list[float] = []
        past_kv = None

        for _ in range(k):
            out = self.model(
                input_ids=input_ids[:, -1:] if past_kv is not None else input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
            tok = logits.argmax().item()
            lp = torch.log_softmax(logits, dim=-1)[tok].item()
            drafted.append(tok)
            log_probs.append(lp)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[tok]], device=self.device)], dim=1
            )

        return MockDraftResponse(
            request_id="", draft_token_ids=drafted, draft_log_probs=log_probs
        )


class LocalTargetEngine:
    """Greedy verification engine with session-based KV caching."""

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._caches: dict[str, tuple[Any, int]] = {}  # session → (past_kv, seq_len)

    @torch.no_grad()
    def verify_drafts(self, req: MockVerifyRequest) -> MockVerifyResponse:
        """Verify draft tokens against the target model (greedy).

        Uses a **full forward pass** over ``prompt + draft`` tokens to
        guarantee numerical parity with ``model.generate()``.  The session
        KV cache is rebuilt afterward to cover only the accepted prefix.
        """
        prompt = req.prompt_token_ids
        drafts = req.draft_token_ids
        full_ids = torch.tensor([prompt + drafts], device=self.device)

        outputs = self.model(full_ids)
        logits = outputs.logits[0]  # (seq_len, vocab)

        # logits[i] predicts position i+1.
        # To verify draft[j] (at position len(prompt)+j), check logits[len(prompt)+j-1].
        n = len(prompt)
        accepted: list[int] = []
        correction: int | None = None

        for j, d in enumerate(drafts):
            target_pred = logits[n + j - 1].argmax().item()
            if target_pred == d:
                accepted.append(d)
            else:
                correction = target_pred
                break

        # If all drafts accepted → bonus token from last logit position
        if len(accepted) == len(drafts) and correction is None:
            bonus = logits[n + len(drafts) - 1].argmax().item()
            accepted.append(bonus)

        # Rebuild session KV cache for the accepted prefix only
        if req.session_id:
            prefix_len = n + len(accepted)
            prefix_ids = torch.tensor(
                [prompt + accepted], device=self.device
            )
            cache_out = self.model(prefix_ids, use_cache=True)
            self._caches[req.session_id] = (
                cache_out.past_key_values,
                prefix_len,
            )

        return MockVerifyResponse(
            request_id=req.request_id,
            accepted_token_ids=accepted,
            correction_token_id=correction,
            num_accepted=len([t for t in accepted if t in drafts[:len(accepted)]]),
            cache_hit=req.session_id in self._caches,
        )

    def end_session(self, session_id: str) -> bool:
        return self._caches.pop(session_id, None) is not None


# =========================================================================
# Mock gRPC stubs — drop-in replacements for the generated stubs
# =========================================================================


class MockDraftServiceStub:
    """Replaces ``spec_decoding_pb2_grpc.DraftServiceStub``.

    Calls :class:`LocalDraftEngine` directly instead of making a network
    round-trip, so the test can run without binding to a real port.
    """

    def __init__(self, engine: LocalDraftEngine) -> None:
        self._engine = engine

    def GenerateDrafts(self, request: MockDraftRequest) -> MockDraftResponse:  # noqa: N802
        return self._engine.generate_drafts(
            prompt_ids=request.prompt_token_ids,
            k=request.max_draft_len,
        )


class MockTargetServiceStub:
    """Replaces ``spec_decoding_pb2_grpc.TargetServiceStub``.

    Calls :class:`LocalTargetEngine` directly.
    """

    def __init__(self, engine: LocalTargetEngine) -> None:
        self._engine = engine

    def VerifyDrafts(self, request: MockVerifyRequest) -> MockVerifyResponse:  # noqa: N802
        return self._engine.verify_drafts(request)


# =========================================================================
# Speculative decoding orchestrator (uses mock stubs)
# =========================================================================


class MockOrchestrator:
    """Full speculative decoding loop wired to mock gRPC stubs.

    Implements the Leviathan et al. (2023) algorithm with greedy sampling:
    draft *k* tokens → verify in one target pass → accept prefix + bonus/
    correction → repeat.
    """

    def __init__(
        self,
        draft_stub: MockDraftServiceStub,
        target_stub: MockTargetServiceStub,
        eos_token_id: int | None = None,
    ) -> None:
        self.draft_stub = draft_stub
        self.target_stub = target_stub
        self.eos_token_id = eos_token_id

    def run(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = MAX_NEW_TOKENS,
        draft_k: int = DRAFT_K,
    ) -> list[int]:
        """Run speculative decoding and return the full token sequence."""
        gen_ids = list(prompt_ids)
        session_id = uuid.uuid4().hex[:16]
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # 1. Draft
            draft_resp = self.draft_stub.GenerateDrafts(
                MockDraftRequest(
                    request_id=uuid.uuid4().hex[:8],
                    prompt_token_ids=gen_ids,
                    max_draft_len=min(draft_k, max_new_tokens - tokens_generated),
                )
            )

            # 2. Verify (mock gRPC call)
            verify_resp = self.target_stub.VerifyDrafts(
                MockVerifyRequest(
                    request_id=draft_resp.request_id,
                    prompt_token_ids=gen_ids,
                    draft_token_ids=draft_resp.draft_token_ids,
                    session_id=session_id,
                )
            )

            # 3. Extend with accepted tokens (includes bonus if all accepted)
            new_tokens = verify_resp.accepted_token_ids
            if verify_resp.correction_token_id is not None:
                new_tokens = list(new_tokens) + [verify_resp.correction_token_id]

            if not new_tokens:
                break  # safety: nothing accepted, nothing corrected

            gen_ids.extend(new_tokens)
            tokens_generated += len(new_tokens)

            # 4. EOS check
            if self.eos_token_id is not None and self.eos_token_id in new_tokens:
                break

        return gen_ids


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model + tokenizer once for the entire test module."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # float32 for CPU determinism
    ).to("cpu").eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture
def mock_stubs(model_and_tokenizer):
    """Build mock gRPC stubs backed by the real model."""
    model, _tok = model_and_tokenizer
    draft_engine = LocalDraftEngine(model)
    target_engine = LocalTargetEngine(model)
    return (
        MockDraftServiceStub(draft_engine),
        MockTargetServiceStub(target_engine),
    )


# =========================================================================
# Helpers
# =========================================================================


def baseline_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Standard greedy ``model.generate()`` — the ground truth."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def specsplit_generate(
    mock_stubs,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    draft_k: int = DRAFT_K,
) -> str:
    """Run speculative decoding via mock-gRPC orchestrator."""
    draft_stub, target_stub = mock_stubs
    orch = MockOrchestrator(
        draft_stub=draft_stub,
        target_stub=target_stub,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_ids = tokenizer.encode(prompt)
    full_ids = orch.run(prompt_ids, max_new_tokens=max_new_tokens, draft_k=draft_k)
    new_ids = full_ids[len(prompt_ids): len(prompt_ids) + max_new_tokens]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# =========================================================================
# Tests
# =========================================================================


class TestExactMatch:
    """Assert speculative decoding output is byte-identical to model.generate()."""

    @pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: p[:40])
    def test_greedy_exact_match(self, model_and_tokenizer, mock_stubs, prompt: str):
        """Spec-decode with same draft/target model must match standard greedy."""
        model, tokenizer = model_and_tokenizer

        baseline = baseline_generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        speculative = specsplit_generate(mock_stubs, tokenizer, prompt, MAX_NEW_TOKENS)

        assert speculative == baseline, (
            f"Output mismatch!\n"
            f"  Baseline:    {baseline!r}\n"
            f"  Speculative: {speculative!r}"
        )

    @pytest.mark.parametrize("k", [1, 3, 5, 10], ids=lambda k: f"k={k}")
    def test_exact_match_varying_draft_depth(
        self, model_and_tokenizer, mock_stubs, k: int
    ):
        """Exact match must hold regardless of draft depth K."""
        model, tokenizer = model_and_tokenizer
        prompt = PROMPTS[0]

        baseline = baseline_generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        speculative = specsplit_generate(
            mock_stubs, tokenizer, prompt, MAX_NEW_TOKENS, draft_k=k
        )

        assert speculative == baseline, (
            f"Mismatch at k={k}!\n"
            f"  Baseline:    {baseline!r}\n"
            f"  Speculative: {speculative!r}"
        )

    def test_short_generation(self, model_and_tokenizer, mock_stubs):
        """Exact match with very short output (max_new_tokens < draft_k)."""
        model, tokenizer = model_and_tokenizer
        prompt = PROMPTS[0]

        baseline = baseline_generate(model, tokenizer, prompt, max_new_tokens=3)
        speculative = specsplit_generate(
            mock_stubs, tokenizer, prompt, max_new_tokens=3, draft_k=5
        )

        assert speculative == baseline

    def test_token_ids_match(self, model_and_tokenizer, mock_stubs):
        """Verify token-level (not just string-level) identity."""
        model, tokenizer = model_and_tokenizer
        prompt = PROMPTS[0]
        prompt_ids = tokenizer.encode(prompt)

        # Baseline token IDs
        input_t = torch.tensor([prompt_ids])
        with torch.no_grad():
            baseline_ids = model.generate(
                input_t, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )[0].tolist()

        # Speculative token IDs
        draft_stub, target_stub = mock_stubs
        orch = MockOrchestrator(
            draft_stub=draft_stub,
            target_stub=target_stub,
            eos_token_id=tokenizer.eos_token_id,
        )
        spec_ids = orch.run(prompt_ids, max_new_tokens=MAX_NEW_TOKENS)

        # Trim to same length
        n = min(len(baseline_ids), len(spec_ids))
        assert baseline_ids[:n] == spec_ids[:n], (
            f"Token ID mismatch at position "
            f"{next(i for i in range(n) if baseline_ids[i] != spec_ids[i])}"
        )


class TestMockGRPCBoundary:
    """Verify the mock stubs faithfully simulate the gRPC boundary."""

    def test_draft_stub_returns_k_tokens(self, mock_stubs):
        draft_stub, _ = mock_stubs
        resp = draft_stub.GenerateDrafts(
            MockDraftRequest(
                request_id="test",
                prompt_token_ids=[1, 2, 3],
                max_draft_len=5,
            )
        )
        assert len(resp.draft_token_ids) == 5
        assert all(isinstance(t, int) for t in resp.draft_token_ids)

    def test_verify_stub_accepts_all_with_same_model(self, mock_stubs):
        """With identical draft/target, all greedy tokens must be accepted."""
        draft_stub, target_stub = mock_stubs
        prompt = [1, 2, 3]
        draft_resp = draft_stub.GenerateDrafts(
            MockDraftRequest(request_id="t", prompt_token_ids=prompt, max_draft_len=3)
        )
        verify_resp = target_stub.VerifyDrafts(
            MockVerifyRequest(
                request_id="t",
                prompt_token_ids=prompt,
                draft_token_ids=draft_resp.draft_token_ids,
                session_id="sess-test",
            )
        )
        # All drafted tokens accepted + 1 bonus
        assert verify_resp.correction_token_id is None
        assert len(verify_resp.accepted_token_ids) == len(draft_resp.draft_token_ids) + 1

    def test_verify_stub_rejects_wrong_tokens(self, mock_stubs):
        """Fabricated tokens should be rejected with a correction."""
        _, target_stub = mock_stubs
        verify_resp = target_stub.VerifyDrafts(
            MockVerifyRequest(
                request_id="t",
                prompt_token_ids=[1, 2, 3],
                draft_token_ids=[999999, 999998, 999997],  # garbage tokens
                session_id="",
            )
        )
        # Very unlikely all garbage tokens match greedy prediction
        assert (
            verify_resp.correction_token_id is not None
            or len(verify_resp.accepted_token_ids) <= 3
        )
