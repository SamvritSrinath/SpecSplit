"""Draft Engine — autoregressive speculative token tree generation.

The ``DraftEngine`` wraps a small, fast language model and generates
speculative token trees of depth *K*. These trees are sent to the Target
Worker for verification.

Architecture Notes:
    - Maintains a local KV cache across rounds to avoid recomputing the
      prefix. The cache is invalidated when the orchestrator signals a
      new prompt or a mismatch.
    - Token trees are represented as nested lists of ``TokenNode``-like
      dicts for easy protobuf conversion.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specsplit.core.config import DraftWorkerConfig
from specsplit.core.serialization import logits_to_probs
from specsplit.core.telemetry import Stopwatch

logger = logging.getLogger(__name__)


@dataclass
class TokenNode:
    """In-memory representation of a single node in the draft token tree."""

    token_id: int
    log_prob: float
    children: list[TokenNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict (mirrors the protobuf ``TokenNode`` message)."""
        return {
            "token_id": self.token_id,
            "log_prob": self.log_prob,
            "children": [c.to_dict() for c in self.children],
        }


class DraftEngine:
    """Autoregressive generation engine for the Draft Worker.

    This class manages model loading, KV cache state, and speculative
    tree generation using a real HuggingFace ``AutoModelForCausalLM``.

    Args:
        config: Draft worker configuration (model name, device, etc.).
    """

    def __init__(self, config: DraftWorkerConfig | None = None) -> None:
        self.config = config or DraftWorkerConfig()
        self.device = torch.device(self.config.device)
        self._model: Any = None  # transformers.AutoModelForCausalLM
        self._tokenizer: Any = None  # transformers.AutoTokenizer
        self._kv_cache: Any = None  # Model-specific past_key_values
        self._cached_prompt_len: int = 0  # Length of prompt encoded in _kv_cache
        self._is_loaded = False

        logger.info(
            "DraftEngine initialized (model=%s, device=%s)",
            self.config.model_name,
            self.config.device,
        )

    def load_model(self) -> None:
        """Load the draft model and tokenizer via ``AutoModelForCausalLM``.

        Uses ``torch.float16`` precision and places the model on the
        configured device.  The tokenizer's pad token is set to
        ``eos_token`` if not already defined.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
        ).to(self.device).eval()

        self._is_loaded = True
        logger.info("Draft model loaded: %s on %s", self.config.model_name, self.device)

    def generate_draft_tree(
        self,
        prompt_ids: list[int],
        k: int | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
    ) -> list[TokenNode]:
        """Generate a speculative token tree from the given prompt context.

        Performs *k* steps of autoregressive generation using KV caching.
        Each "beam" is an independent greedy/sampled chain (no beam-search
        coupling).  The result is a list of root ``TokenNode`` objects,
        each heading a flat chain of depth *k*.

        Args:
            prompt_ids: Tokenized prompt (list of vocabulary indices).
            k: Tree depth (defaults to ``config.max_draft_tokens``).
            num_beams: Number of independent chains (defaults to
                ``config.num_beams``).
            temperature: Sampling temperature (defaults to
                ``config.temperature``).

        Returns:
            A list of root-level ``TokenNode`` objects forming the draft
            forest.
        """
        k = k or self.config.max_draft_tokens
        num_beams = num_beams or self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature

        sw = Stopwatch()
        sw.start()

        if not self._is_loaded or self._model is None:
            raise RuntimeError("Draft model not loaded. Call load_model() first.")

        # -----------------------------------------------------------------
        # Encode the prompt prefix (or reuse cached KV)
        # -----------------------------------------------------------------
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        # Shape: [1, prompt_len]

        with torch.no_grad():
            # Always recompute prompt for simplicity per-call
            # (cross-round caching will be added in a future PR)
            prefix_out = self._model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )
            past_kv = prefix_out.past_key_values
            # last_logits shape: [1, prompt_len, vocab_size]
            last_logits = prefix_out.logits[:, -1, :]
            # Shape: [1, vocab_size]

        # -----------------------------------------------------------------
        # Autoregressive draft generation — produce `num_beams` chains
        # -----------------------------------------------------------------
        roots: list[TokenNode] = []

        for _beam_idx in range(num_beams):
            # Each beam starts from the same prompt prefix KV cache
            beam_past_kv = past_kv
            current_logits = last_logits  # Shape: [1, vocab_size]
            chain: list[tuple[int, float]] = []

            for _step in range(k):
                with torch.no_grad():
                    # Sample or greedy-pick the next token
                    probs = logits_to_probs(current_logits, temperature=temperature)
                    # Shape: [1, vocab_size]

                    if temperature == 0.0:
                        # Greedy
                        next_token_id = current_logits.argmax(dim=-1)  # [1]
                    else:
                        # Top-k=1 multinomial (temperature-scaled greedy for
                        # temp≈1, or stochastic sampling)
                        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        # Shape: [1]

                    token_id_int = next_token_id.item()
                    # Log probability of the chosen token
                    log_prob = math.log(
                        probs[0, token_id_int].item() + 1e-10
                    )
                    chain.append((token_id_int, log_prob))

                    # Forward pass for the next step with KV cache
                    next_input = next_token_id.unsqueeze(0)  # [1, 1]
                    step_out = self._model(
                        input_ids=next_input,
                        past_key_values=beam_past_kv,
                        use_cache=True,
                    )
                    beam_past_kv = step_out.past_key_values
                    current_logits = step_out.logits[:, -1, :]
                    # Shape: [1, vocab_size]

            # Build a flat TokenNode chain from the collected tokens
            if chain:
                root = TokenNode(token_id=chain[0][0], log_prob=chain[0][1])
                current_node = root
                for tid, lp in chain[1:]:
                    child = TokenNode(token_id=tid, log_prob=lp)
                    current_node.children.append(child)
                    current_node = child
                roots.append(root)

        sw.stop()
        logger.debug(
            "Generated draft tree: depth=%d, beams=%d, elapsed=%.3f ms",
            k,
            num_beams,
            sw.elapsed_ms,
        )
        return roots

    def reset_cache(self) -> None:
        """Clear the KV cache (e.g., on prompt change or verification failure)."""
        self._kv_cache = None
        self._cached_prompt_len = 0
        logger.debug("KV cache cleared")

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._is_loaded
