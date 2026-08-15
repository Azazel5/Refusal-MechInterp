"""Model backend abstraction.

Step 0 (behavioral measurement) only needs `generate()`. Steps 1-3 (path patching,
intervention, SAE feature checks) need activation/hook access, which in practice
means running via TransformerLens/nnsight/SAELens against local or remote GPU
weights — there's no clean way to do that through a hosted chat API, so
`get_activations` / `apply_hooks` are only implemented on the local backend.

Nothing here is wired to real weights yet — this repo has no GPU/model access
configured (see docs/BRIEF.md revision log, 2026-08-15). Fill in `HFLocalBackend`
once compute is available; `NotImplementedError` is intentional, not a bug.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass
class GenerationResult:
    prompt: str
    completion: str
    model_id: str
    raw: Any = None


class ModelBackend(abc.ABC):
    """Minimum interface Step 0 needs."""

    model_id: str

    @abc.abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int = 256, **kwargs) -> GenerationResult:
        ...


class ActivationCapableBackend(ModelBackend):
    """Extended interface Steps 1-3 need: cached activations and hook-based
    patching/ablation, in the TransformerLens/nnsight sense."""

    @abc.abstractmethod
    def run_with_cache(self, prompt: str) -> tuple[GenerationResult, dict[str, Any]]:
        """Return the generation plus a cache of intermediate activations keyed by
        hook name (e.g. 'blocks.{layer}.hook_resid_post')."""
        ...

    @abc.abstractmethod
    def run_with_hooks(
        self,
        prompt: str,
        hooks: list[tuple[str, Callable]],
    ) -> GenerationResult:
        """Run with the given (hook_name, hook_fn) pairs applied — the mechanism
        for activation patching / ablation."""
        ...


class HFLocalBackend(ActivationCapableBackend):
    """Local weights via transformer_lens/nnsight + SAELens. NOT IMPLEMENTED —
    this is the integration point once GPU access + model weights are available.

    Expected libraries (see requirements.txt): transformer_lens, sae_lens, nnsight.
    """

    def __init__(self, model_id: Literal["gemma-3-27b-it", "llama-3.3-70b-it"], device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        raise NotImplementedError(
            "HFLocalBackend requires GPU + model weights, not configured in this "
            "environment yet. See docs/BRIEF.md and configs/models.yaml."
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 256, **kwargs) -> GenerationResult:
        raise NotImplementedError

    def run_with_cache(self, prompt: str):
        raise NotImplementedError

    def run_with_hooks(self, prompt: str, hooks):
        raise NotImplementedError


class ChatAPIBackend(ModelBackend):
    """Generation-only backend via a hosted chat completions API (OpenRouter,
    Together, Fireworks, etc.) — usable for Step 0 baseline measurement without
    GPU access, but cannot support Steps 1-3 (no activation access through a
    chat API). NOT WIRED to a real key yet; set api_key/base_url/model_id and
    implement the request in `generate()` once you've chosen a provider.
    """

    def __init__(self, model_id: str, api_key: str, base_url: str):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str, *, max_new_tokens: int = 256, **kwargs) -> GenerationResult:
        raise NotImplementedError(
            "Wire this up to your chosen provider's chat completions endpoint. "
            "Kept unimplemented rather than guessed at, since the provider/keys "
            "haven't been chosen yet."
        )
