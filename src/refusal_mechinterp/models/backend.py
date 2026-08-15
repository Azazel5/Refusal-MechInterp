"""Model backend abstraction.

Step 0 (behavioral measurement) only needs `generate()`. Steps 1-3 (path patching,
intervention, SAE feature checks) need activation/hook access — `HFLocalBackend`
below implements that via TransformerLens (`transformer_lens.HookedTransformer`),
per docs/MODEL_ACQUISITION.md section 5's guidance that TL has the most mature hook
API for Gemma/Llama-family models. Its hook-naming convention
(`blocks.{layer}.hook_resid_post` etc.) is what refusal_direction.py and
run_path_patching.py already assume.

**Caveat, unresolved as of 2026-08-15:** TransformerLens's support matrix for Gemma 3
27B-IT and Llama 3.3 70B-IT specifically has not been verified against a real install
(no GPU in this environment) — docs/MODEL_ACQUISITION.md section 5 explicitly warns
not to assume TL/nnsight support both architectures equally without checking. If
`HookedTransformer.from_pretrained(hf_id)` fails for either model, fall back to the
`engine="nnsight"` path, which is stubbed (NotImplementedError) rather than guessed
at, since its actual API shape for cache/hook access should be written against
whichever nnsight version is installed at that point, not assumed here.

Quantization: 4-bit is fine for `generate()`-only Step 0 screening (see
scripts/download_models.py) but NOT valid for activation work — `HFLocalBackend`
below defaults to bf16 and only accepts `quantize="4bit"` when
`activation_capable=False` is also passed, to make that constraint load-bearing
rather than a comment someone can miss.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Literal

MODEL_HF_IDS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "llama-3.3-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
}


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
    """Local weights via TransformerLens (default) or nnsight (fallback, stubbed).

    Not runnable in this environment (no GPU / transformer_lens / model weights
    here) — this is the real implementation for use on Colab/HPC per
    docs/MODEL_ACQUISITION.md, written precisely enough to be a run-it-and-debug
    job rather than a design job once you have compute.
    """

    def __init__(
        self,
        model_id: Literal["gemma-3-27b-it", "llama-3.3-70b-it"],
        device: str = "cuda",
        engine: Literal["transformer_lens", "nnsight"] = "transformer_lens",
        quantize: Literal["4bit", "bf16"] | None = None,
        activation_capable: bool = True,
    ):
        if quantize == "4bit" and activation_capable:
            raise ValueError(
                "4-bit quantization + activation_capable=True is refused on purpose "
                "(docs/MODEL_ACQUISITION.md section 3: quantized activations are not "
                "a valid substrate for path patching / refusal-direction fidelity). "
                "Pass activation_capable=False for a Step-0-only quantized backend, "
                "or quantize='bf16'/None for Steps 1-3."
            )
        if model_id not in MODEL_HF_IDS:
            raise ValueError(f"Unknown model_id {model_id!r}, expected one of {list(MODEL_HF_IDS)}")

        self.model_id = model_id
        self.hf_id = MODEL_HF_IDS[model_id]
        self.device = device
        self.engine = engine
        self.activation_capable = activation_capable

        if engine == "nnsight":
            raise NotImplementedError(
                "nnsight engine path not implemented — write it against whichever "
                "nnsight version is installed once you've confirmed TransformerLens "
                "doesn't load this checkpoint cleanly (docs/MODEL_ACQUISITION.md "
                "section 5). transformer_lens is the default and primary path."
            )

        self._load_transformer_lens(quantize or "bf16")

    def _load_transformer_lens(self, quantize: str) -> None:
        try:
            import torch
            from transformer_lens import HookedTransformer
        except ImportError as e:
            raise ImportError(
                "transformer_lens (and torch) not installed. "
                "`pip install transformer_lens torch` first — see docs/MODEL_ACQUISITION.md section 5."
            ) from e

        dtype = torch.bfloat16 if quantize == "bf16" else None
        # HookedTransformer.from_pretrained's own kwargs for quantized loading vary
        # by TL version — if this errors, check the installed TL version's docs for
        # its current 4-bit/bitsandbytes integration rather than guessing here.
        load_kwargs: dict[str, Any] = {"device": self.device}
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        print(f"Loading {self.hf_id} via TransformerLens (this can take a while for 27B/70B)...")
        self.model = HookedTransformer.from_pretrained(self.hf_id, **load_kwargs)
        self.tokenizer = self.model.tokenizer

    def _apply_chat_template(self, prompt: str) -> str:
        # Gemma and Llama use different chat-template conventions (docs/
        # MODEL_ACQUISITION.md section 6) — always go through the tokenizer's own
        # template rather than hand-rolling the prompt string.
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 256, **kwargs) -> GenerationResult:
        templated = self._apply_chat_template(prompt)
        tokens = self.model.to_tokens(templated)
        output_tokens = self.model.generate(
            tokens, max_new_tokens=max_new_tokens, do_sample=False, **kwargs
        )
        completion_tokens = output_tokens[0, tokens.shape[1]:]
        completion = self.model.to_string(completion_tokens)
        return GenerationResult(prompt=prompt, completion=completion, model_id=self.model_id)

    def run_with_cache(self, prompt: str) -> tuple[GenerationResult, dict[str, Any]]:
        if not self.activation_capable:
            raise RuntimeError("Backend loaded with activation_capable=False.")
        templated = self._apply_chat_template(prompt)
        tokens = self.model.to_tokens(templated)
        logits, cache = self.model.run_with_cache(tokens)
        completion = self.model.to_string(logits[0, -1:].argmax(dim=-1))
        result = GenerationResult(prompt=prompt, completion=completion, model_id=self.model_id, raw=logits)
        return result, cache

    def run_with_hooks(self, prompt: str, hooks: list[tuple[str, Callable]]) -> GenerationResult:
        if not self.activation_capable:
            raise RuntimeError("Backend loaded with activation_capable=False.")
        templated = self._apply_chat_template(prompt)
        tokens = self.model.to_tokens(templated)
        logits = self.model.run_with_hooks(tokens, fwd_hooks=hooks)
        completion = self.model.to_string(logits[0, -1:].argmax(dim=-1))
        return GenerationResult(prompt=prompt, completion=completion, model_id=self.model_id, raw=logits)


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
