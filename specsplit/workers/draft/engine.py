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
from dataclasses import dataclass, field
from typing import Any

import torch

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
    tree generation. The actual model inference is stubbed with TODO
    markers — plug in your model-specific forward pass.

    Args:
        config: Draft worker configuration (model name, device, etc.).
    """

    def __init__(self, config: DraftWorkerConfig | None = None) -> None:
        self.config = config or DraftWorkerConfig()
        self.device = torch.device(self.config.device)
        self._model: Any = None  # Will hold the HF model
        self._tokenizer: Any = None  # Will hold the HF tokenizer
        self._kv_cache: Any = None  # Model-specific KV cache
        self._is_loaded = False

        logger.info(
            "DraftEngine initialized (model=%s, device=%s)",
            self.config.model_name,
            self.config.device,
        )

    def load_model(self) -> None:
        """Load the draft model and tokenizer.

        .. todo::
            Replace the stub below with actual ``AutoModelForCausalLM`` /
            ``AutoTokenizer`` loading from HuggingFace.
        """
        # TODO(model-loading): Uncomment and adapt for your model:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     self.config.model_name,
        #     torch_dtype=torch.float16,
        # ).to(self.device).eval()
        self._is_loaded = True
        logger.info("Draft model loaded: %s", self.config.model_name)

    def generate_draft_tree(
        self,
        prompt_ids: list[int],
        k: int | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
    ) -> list[TokenNode]:
        """Generate a speculative token tree from the given prompt context.

        Performs *k* steps of autoregressive generation, optionally branching
        at each step to produce ``num_beams`` children per node.

        Args:
            prompt_ids: Tokenized prompt (list of vocabulary indices).
            k: Tree depth (defaults to ``config.max_draft_tokens``).
            num_beams: Branching factor (defaults to ``config.num_beams``).
            temperature: Sampling temperature (defaults to ``config.temperature``).

        Returns:
            A list of root-level ``TokenNode`` objects forming the draft forest.

        .. todo::
            Wire up the real model forward pass and KV cache management.
        """
        k = k or self.config.max_draft_tokens
        num_beams = num_beams or self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature

        sw = Stopwatch()
        sw.start()

        # TODO(inference): Replace this stub with actual model inference.
        # The stub generates a linear chain of dummy tokens for testing.
        roots: list[TokenNode] = []
        for beam in range(num_beams):
            node = TokenNode(token_id=beam + 1, log_prob=-0.1 * (beam + 1))
            current = node
            for depth in range(1, k):
                child = TokenNode(
                    token_id=(beam + 1) * 100 + depth,
                    log_prob=-0.1 * (depth + 1),
                )
                current.children.append(child)
                current = child
            roots.append(node)

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
        logger.debug("KV cache cleared")

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._is_loaded
