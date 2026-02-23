"""Draft Engine — autoregressive speculative token tree generation.

The ``DraftEngine`` wraps a small, fast language model and generates
speculative token trees of depth *K*. These trees are sent to the Target
Worker for verification.

Architecture Notes:
    - Cross-round KV cache reuse is planned but not yet implemented;
      each call currently recomputes the full prompt prefix. The
      ``_kv_cache`` / ``reset_cache()`` plumbing is retained for the
      future implementation.
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
        self._cached_last_logits: torch.Tensor | None = None  # Logits at end of cached prefix
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

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
            )
            .to(self.device)
            .eval()
        )

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
        # Encode the prompt prefix (with cross-round KV cache reuse)
        # -----------------------------------------------------------------
        new_ids = prompt_ids[self._cached_prompt_len:]

        if self._kv_cache is not None and len(prompt_ids) >= self._cached_prompt_len:
            if new_ids:
                # Case B: context grew — extend from cached KV with only the new tokens
                new_input = torch.tensor([new_ids], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    prefix_out = self._model(
                        input_ids=new_input,
                        past_key_values=self._kv_cache,
                        use_cache=True,
                    )
                past_kv = prefix_out.past_key_values
                last_logits = prefix_out.logits[:, -1, :]
            else:
                # Case C: exact match — reuse stored KV and logits, no forward pass needed
                past_kv = self._kv_cache
                last_logits = self._cached_last_logits
        else:
            # Case A: no cache or stale — full prefix recompute from scratch
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                prefix_out = self._model(
                    input_ids=input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
            past_kv = prefix_out.past_key_values
            last_logits = prefix_out.logits[:, -1, :]

        # Update cache state for next round
        self._kv_cache = past_kv
        self._cached_prompt_len = len(prompt_ids)
        self._cached_last_logits = last_logits

        # -----------------------------------------------------------------
        # Task 2.2: True Tree Construction (BFS Level-Order Expansion)
        # -----------------------------------------------------------------
        roots: list[TokenNode] = []

        # Queue stores: (parent_node, past_key_values, logits, current_depth)
        # We start with the prompt's output state. parent_node=None means these will be roots.
        queue = [(None, past_kv, last_logits, 0)]
        
        while queue:
            parent_node, past_kv, logits, depth = queue.pop(0)
            
            if depth >= k:
                continue
                
            # Dynamic Branching Topology: 
            # To prevent exponential explosion (e.g., 2^5 = 32 nodes), we branch heavily
            # at the root (depth 0), and reduce branching deeper in the tree.
            # E.g., if num_beams=3: Depth 0 branches 3 ways, Depth 1 branches 2 ways, Depth 2+ is straight lines.
            branching_factor = max(1, num_beams - depth)
            
            with torch.no_grad():
                probs = logits_to_probs(logits, temperature=temperature)
                
                if temperature == 0.0:
                    # Top-B greedy probabilities
                    top_probs, top_indices = torch.topk(probs, branching_factor, dim=-1)
                else:
                    # Top-B stochastic sampling
                    top_indices = torch.multinomial(probs, num_samples=branching_factor)
                    top_probs = torch.gather(probs, -1, top_indices)
                
                # Expand each selected branch
                for i in range(branching_factor):
                    token_id_int = top_indices[0, i].item()
                    log_prob = math.log(top_probs[0, i].item() + 1e-10)
                    
                    child_node = TokenNode(token_id=token_id_int, log_prob=log_prob)
                    
                    if parent_node is None:
                        roots.append(child_node)
                    else:
                        parent_node.children.append(child_node)
                        
                    # Compute the forward pass for this specific branch
                    next_input = torch.tensor([[token_id_int]], dtype=torch.long, device=self.device)
                    step_out = self._model(
                        input_ids=next_input,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    
                    # Add child to the queue to expand next depth
                    queue.append((
                        child_node, 
                        step_out.past_key_values, 
                        step_out.logits[:, -1, :], 
                        depth + 1
                    ))

        sw.stop()
        logger.debug(
            "Generated true draft tree: depth=%d, root_branching=%d, elapsed=%.3f ms",
            k,
            num_beams,
            sw.elapsed_ms,
        )
        return roots

    def reset_cache(self) -> None:
        """Clear the KV cache (e.g., on prompt change or verification failure)."""
        self._kv_cache = None
        self._cached_prompt_len = 0
        self._cached_last_logits = None
        logger.debug("KV cache cleared")

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._is_loaded
