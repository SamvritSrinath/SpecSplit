"""Unit tests for specsplit.core.serialization."""

from __future__ import annotations

import pytest
import torch

from specsplit.core.serialization import logits_to_probs, tensor_to_token_ids, token_ids_to_tensor


class TestTensorToTokenIds:
    """Tests for tensor → list conversion."""

    def test_basic_roundtrip(self, sample_token_ids: list[int], sample_tensor: torch.Tensor):
        """Converting tensor → list should return the original IDs."""
        result = tensor_to_token_ids(sample_tensor)
        assert result == sample_token_ids

    def test_empty_tensor(self):
        """Empty tensor should produce an empty list."""
        t = torch.tensor([], dtype=torch.long)
        assert tensor_to_token_ids(t) == []

    def test_rejects_2d_tensor(self):
        """Should raise ValueError for multi-dimensional tensors."""
        t = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1-D tensor"):
            tensor_to_token_ids(t)

    def test_gpu_tensor_returns_cpu_list(self):
        """Tensor on any device should produce a CPU Python list."""
        t = torch.tensor([42, 99], dtype=torch.long)
        result = tensor_to_token_ids(t)
        assert isinstance(result, list)
        assert result == [42, 99]


class TestTokenIdsToTensor:
    """Tests for list → tensor conversion."""

    def test_basic_conversion(self, sample_token_ids: list[int]):
        """List of ints should become a 1-D LongTensor."""
        t = token_ids_to_tensor(sample_token_ids)
        assert t.shape == (len(sample_token_ids),)
        assert t.dtype == torch.long

    def test_empty_list(self):
        """Empty list should produce a 0-length tensor."""
        t = token_ids_to_tensor([])
        assert t.shape == (0,)

    def test_custom_dtype(self):
        """Should respect a custom dtype argument."""
        t = token_ids_to_tensor([1, 2, 3], dtype=torch.int32)
        assert t.dtype == torch.int32

    def test_roundtrip(self, sample_token_ids: list[int]):
        """list → tensor → list should be identity."""
        t = token_ids_to_tensor(sample_token_ids)
        result = tensor_to_token_ids(t)
        assert result == sample_token_ids


class TestLogitsToProbs:
    """Tests for logits → probability conversion."""

    def test_output_sums_to_one(self, sample_logits: torch.Tensor):
        """Probabilities should sum to 1.0."""
        probs = logits_to_probs(sample_logits)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_temperature_scaling(self, sample_logits: torch.Tensor):
        """Higher temperature should produce a flatter distribution."""
        probs_low = logits_to_probs(sample_logits, temperature=0.1)
        probs_high = logits_to_probs(sample_logits, temperature=10.0)
        # Low temp → sharper → higher max prob
        assert probs_low.max() > probs_high.max()

    def test_greedy_temperature_zero(self, sample_logits: torch.Tensor):
        """Temperature 0 should produce a one-hot distribution."""
        probs = logits_to_probs(sample_logits, temperature=0)
        assert probs.max().item() == pytest.approx(1.0)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert probs.argmax() == sample_logits.argmax()

    def test_negative_temperature_raises(self, sample_logits: torch.Tensor):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            logits_to_probs(sample_logits, temperature=-1.0)
