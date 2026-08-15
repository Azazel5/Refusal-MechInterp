"""SAE / transcoder loading via SAELens, per docs/MODEL_ACQUISITION.md section 4.

Always load through the library (`SAE.from_pretrained`), never fetch weight files
by hand — that keeps config/hook-point metadata correct. Release strings below are
placeholders flagged for verification; run scripts/verify_repo_ids.py --check-saelens
before trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SAESpec:
    release: str
    sae_id_template: str  # e.g. "layer_{layer}/width_16k/l0_medium"
    verified: bool = False


# NOT VERIFIED — run scripts/verify_repo_ids.py --check-saelens against a real
# sae_lens install before using these. Placeholder strings from
# docs/MODEL_ACQUISITION.md section 4.
GEMMA_SCOPE_2_RES = SAESpec(
    release="gemma-scope-2-27b-it-res",
    sae_id_template="layer_{layer}/width_16k/l0_medium",
)
GEMMA_SCOPE_2_TRANSCODER = SAESpec(
    release="gemma-scope-2-27b-it-transcoder",
    sae_id_template="layer_{layer}/width_16k/l0_medium",
)
LLAMA_3_3_70B_SAE = SAESpec(
    release="llama-3.3-70b-sae",  # exact string NOT confirmed, see docs/MODEL_ACQUISITION.md
    sae_id_template="layer_{layer}",
)


def load_sae(spec: SAESpec, layer: int, device: str = "cuda"):
    """Load an SAE at the given layer. Layer choice should come from Step 1's
    localization, not be fixed in advance (per docs/MODEL_ACQUISITION.md section 4).

    Requires `sae_lens` installed and (for gated releases) HF access configured.
    Not runnable in this environment — no GPU/sae_lens here.
    """
    try:
        from sae_lens import SAE
    except ImportError as e:
        raise ImportError("sae_lens not installed. `pip install sae-lens` first.") from e

    if not spec.verified:
        import warnings
        warnings.warn(
            f"SAESpec.release={spec.release!r} has not been verified against a "
            "live sae_lens pretrained_saes_directory() — run "
            "scripts/verify_repo_ids.py --check-saelens first and set verified=True "
            "once confirmed correct, so this warning stops firing.",
            stacklevel=2,
        )

    sae_id = spec.sae_id_template.format(layer=layer)
    sae, cfg, sparsity = SAE.from_pretrained(release=spec.release, sae_id=sae_id, device=device)
    return sae, cfg, sparsity
