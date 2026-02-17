"""Target Engine — stateless tree-attention verification.

The ``TargetEngine`` wraps the large, accurate language model. Given a draft
token tree from the Draft Worker, it performs a single batched forward pass
using tree attention to verify which draft tokens are accepted under the
target distribution.

Architecture Notes:
    - **Stateless**: No KV cache is maintained between calls. Each
      verification is independent, making the Target Worker horizontally
      scalable.
    - Uses rejection sampling: for each position in the tree, if
      ``p_target(x) >= p_draft(x)``, the token is accepted; otherwise
      it is accepted with probability ``p_target(x) / p_draft(x)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from specsplit.core.config import TargetWorkerConfig
from specsplit.core.telemetry import Stopwatch

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a draft tree against the target model."""

    accepted_token_ids: list[int]
    correction_token_id: int | None
    num_accepted: int

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted (0.0–1.0)."""
        total = self.num_accepted + (1 if self.correction_token_id is not None else 0)
        return self.num_accepted / total if total > 0 else 0.0


class TargetEngine:
    """Tree-attention verification engine for the Target Worker.

    This engine is stateless — each call to ``verify_draft_tree`` is
    independent and does not rely on cached state from prior calls.

    Args:
        config: Target worker configuration (model name, device, etc.).
    """

    def __init__(self, config: TargetWorkerConfig | None = None) -> None:
        self.config = config or TargetWorkerConfig()
        self.device = torch.device(self.config.device)
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_loaded = False

        logger.info(
            "TargetEngine initialized (model=%s, device=%s)",
            self.config.model_name,
            self.config.device,
        )

    def load_model(self) -> None:
        """Load the target model and tokenizer.

        .. todo::
            Replace the stub below with actual model loading.
        """
        # TODO(model-loading): Uncomment and adapt:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     self.config.model_name,
        #     torch_dtype=torch.float16,
        # ).to(self.device).eval()
        self._is_loaded = True
        logger.info("Target model loaded: %s", self.config.model_name)

    def verify_draft_tree(
        self,
        prompt_ids: list[int],
        draft_tree: list[dict[str, Any]],
    ) -> VerificationResult:
        """Verify a draft token tree against the target model's distribution.

        Performs a single batched forward pass with tree attention to score
        all candidate paths simultaneously, then applies rejection sampling
        to determine the longest accepted prefix.

        Args:
            prompt_ids: Original prompt token IDs.
            draft_tree: Draft tree as a list of dicts (from ``TokenNode.to_dict()``).
                Each dict has keys ``token_id``, ``log_prob``, ``children``.

        Returns:
            A ``VerificationResult`` with accepted tokens and an optional
            correction token.

        .. todo::
            Implement tree-attention forward pass and rejection sampling.
        """
        sw = Stopwatch()
        sw.start()

        # TODO(verification): Replace this stub with actual tree-attention
        # verification. The stub accepts all tokens from the first path.
        accepted: list[int] = []
        node_list = draft_tree
        while node_list:
            node = node_list[0]  # Follow the first branch
            accepted.append(node["token_id"])
            node_list = node.get("children", [])

        # Stub: no correction needed (all accepted)
        correction = None

        sw.stop()
        logger.debug(
            "Verification complete: accepted=%d tokens, elapsed=%.3f ms",
            len(accepted),
            sw.elapsed_ms,
        )

        return VerificationResult(
            accepted_token_ids=accepted,
            correction_token_id=correction,
            num_accepted=len(accepted),
        )

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._is_loaded
