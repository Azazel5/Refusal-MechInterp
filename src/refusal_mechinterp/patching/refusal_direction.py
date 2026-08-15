"""Refusal direction — Arditi et al. (2024) mean-difference method.

direction = mean(activations on harmful prompts) - mean(activations on harmless prompts)

computed at a chosen layer's residual stream, at the last token position, averaged
over a paired harmful/harmless prompt set (here: tier (c) vs tier (a) items, matched
by triple, is the natural pairing for this project rather than a generic harmful/
harmless corpus — using the matched triples keeps the direction on-topic with what
Step 1 is probing).

Blocked on ActivationCapableBackend (see models/backend.py) — not runnable yet.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from refusal_mechinterp.models.backend import ActivationCapableBackend


def compute_refusal_direction(
    backend: "ActivationCapableBackend",
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer: int,
    hook_name_template: str = "blocks.{layer}.hook_resid_post",
):
    """Returns a single direction vector (torch.Tensor) at the given layer.

    NOT IMPLEMENTED: requires `run_with_cache` on a real backend. Left as a
    precise stub — the math is one line once activations are available:

        harmful_acts = stack([cache[hook_name][:, -1, :] for cache in harmful_caches])
        harmless_acts = stack([cache[hook_name][:, -1, :] for cache in harmless_caches])
        direction = harmful_acts.mean(0) - harmless_acts.mean(0)
        return direction / direction.norm()
    """
    raise NotImplementedError(
        "Requires a real ActivationCapableBackend (GPU + TransformerLens/SAELens). "
        "See docstring for the exact computation to run once available."
    )
