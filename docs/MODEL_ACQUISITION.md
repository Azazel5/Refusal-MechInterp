# Model Acquisition Instructions — Gemma 3 27B-IT & Llama 3.3 70B-IT

**For: Colab (interactive/pilot) and HPC (full runs). Companion to configs/models.yaml.**

> Pasted verbatim 2026-08-15. Repo-ID verification performed live where possible
> (curl against the HF API, no HF_TOKEN needed for basic metadata) — see
> "Implementation status" at the end for what was actually checked vs. still needs
> checking once compute/tokens are available.

---

## 0. Before anything else: verify exact repo IDs

HF repo IDs and SAE release IDs shift over time — the existing
`fetch_seed_datasets.py` docstring already flags this for XSTest. Do not
hardcode IDs from this doc into `configs/models.yaml` without a live check:

```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); \
print([m.id for m in api.list_models(search='gemma-3-27b', limit=10)])"
```
Cross-check SAE release IDs against SAELens's own pretrained-SAE table rather
than guessing:
```bash
python -c "from sae_lens import pretrained_saes_directory; \
print([k for k in pretrained_saes_directory() if 'gemma' in k.lower()])"
```
and against Neuronpedia's model API (`https://www.neuronpedia.org/api/models`)
for the Llama-side SAE/NLA layer numbers.

---

## 1. Access requirements (do this first — gated repos)

Both base models are gated on Hugging Face. Before any download will succeed:

1. Create/verify an HF account, generate a token with **read** scope.
2. Visit each model page while logged in and accept the license:
   - `google/gemma-3-27b-it` (Google usage license)
   - `meta-llama/Llama-3.3-70B-Instruct` (Meta Llama license)
3. Set the token as an environment variable, never hardcode it:
   ```bash
   export HF_TOKEN=hf_xxxxxxxx
   huggingface-cli login --token $HF_TOKEN
   ```
4. On Colab, store it as a Colab secret (`userdata.get('HF_TOKEN')`), not in
   the notebook body, so it doesn't end up committed if the notebook is saved.

---

## 2. Hardware reality check — pick the right platform per model

| Model | Params | bf16 size | 4-bit size | Fits on Colab? |
|---|---|---|---|---|
| Gemma 3 27B-IT | 27B | ~54 GB | ~15–17 GB | Colab Pro/Pro+ with A100 (40GB) — yes, in 4-bit. Free-tier T4 (16GB) — too tight, will OOM with any activation caching for path patching. |
| Llama 3.3 70B-IT | 70B | ~140 GB | ~38–40 GB | No single Colab GPU tier fits this comfortably even in 4-bit once you add activation caching / hook overhead for Steps 1–3. Treat as **HPC-only**. |

**Recommendation:** use Colab for Gemma 3 27B-IT pilot work (Step 0 baseline,
early Step 1 dry-runs) since it's the better-instrumented model anyway (full
per-layer Gemma Scope 2 SAE coverage). Reserve HPC allocation for Llama 3.3
70B-IT and for any full-scale run of both models together.

---

## 3. Downloading the base models

**Weights only (generation, Step 0 baseline behavioral runs):**
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-3-27b-it",
    local_dir="models/gemma-3-27b-it",
    token=True,          # picks up HF_TOKEN from env / huggingface-cli login
)
```
Same pattern for `meta-llama/Llama-3.3-70B-Instruct` on HPC — don't attempt
the full 140GB snapshot on Colab's disk quota, stream via `from_pretrained`
with `device_map="auto"` + quantization instead (see below).

**Quantized load for Colab (Gemma 3 27B-IT, 4-bit via bitsandbytes):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it", token=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it",
    quantization_config=bnb_config,
    device_map="auto",
    token=True,
)
```

**Caution on quantization for Steps 1–3:** 4-bit quantization distorts
activations enough to matter for path patching / refusal-direction fidelity.
Use 4-bit only for Step 0 behavioral screening on Colab. For anything that
reads or patches internal activations (Steps 1, 2, 3, Q6/Q7), run in bf16 on
HPC — quantized activations are not a valid substrate for the causal claims
this project is making.

---

## 4. Pulling the interpretability stack (SAEs / transcoders / NLA)

**Gemma Scope 2 (Gemma 3 27B-IT) via SAELens — do not manually download SAE
weight files, load through the library so config/hook-point metadata stays
correct:**
```python
from sae_lens import SAE

sae, cfg, sparsity = SAE.from_pretrained(
    release="gemma-scope-2-27b-it-res",   # VERIFY exact release string per Sec. 0
    sae_id="layer_40/width_16k/l0_medium", # VERIFY per Sec. 0 — layer choice should
                                            # match Step 1's localized layer(s), not
                                            # be fixed in advance
    device="cuda",
)
```
Transcoders / cross-layer transcoders (for path-patching attribution across
layers, per Step 1's methodology) are pulled the same way with
`release="gemma-scope-2-27b-it-transcoder"` (verify exact string).

**Llama 3.3 70B-IT SAEs:** confirmed to exist on Neuronpedia as of this
project's preliminary research, but the exact HF release ID was not verified
in this pass — run the Sec. 0 check against SAELens's directory before
building `configs/models.yaml` around a guessed string.

**NLA (Natural Language Autoencoders):** Gemma 3 27B-IT at layer 41, Llama
3.3 70B-IT at layer 53 — these layer numbers come from the existing
Neuronpedia NLA tutorial pairing (Fraser-Taliente, Kantamneni, Ong et al.
2026) already used to justify this model pair in docs/BRIEF.md Sec. 2. Use
these as your starting hook points for J-Lens, not the whole depth — verify
they land inside whatever range Step 1's path patching identifies as causal
before trusting them as final.

---

## 5. Environment setup (Colab and HPC)

```bash
pip install --break-system-packages \
    transformers accelerate bitsandbytes \
    sae-lens transformer-lens nnsight \
    datasets sentence-transformers  # sentence-transformers needed by
                                     # scripts/resolve_tier_c.py
```
On HPC, prefer a proper venv/conda env over `--break-system-packages`; on
Colab the flag (or an equivalent `!pip install`) is fine since the runtime is
disposable.

**Path patching / activation hooks (Step 1):** `transformer_lens` has the
most mature hook API for Gemma/Llama-family models; `nnsight` is the fallback
Neuronpedia itself now uses for broader model support — pick one per model
based on which loads Gemma 3 27B-IT / Llama 3.3 70B-IT cleanly, don't assume
both libraries support both architectures equally without checking.

---

## 6. Sanity check before any real run

Before pointing Step 0 at the full `data/matched_triples/` or
`data/conversations/turn_split_conversations.jsonl`, run a single forward
pass on one benign and one control-risk item on each platform and confirm:
- Generation completes without OOM.
- Tokenizer chat template matches what each model's IT/instruct format
  expects (Gemma and Llama use different chat-template conventions —
  verify with `tok.apply_chat_template(...)`, don't hand-roll the prompt
  string).
- SAE/NLA hook points actually fire on a real forward pass (a silent
  shape-mismatch here will otherwise surface much later, mid-Step-1).

---

## Implementation status (2026-08-15)

- **Repo ID discrepancy caught and resolved.** The user's pasted links included
  `meta-llama/Meta-Llama-3-70B` (base, non-instruct, Llama 3 not 3.3) alongside this
  doc's `meta-llama/Llama-3.3-70B-Instruct`. Live-checked both against the HF API
  (`curl .../api/models/<id>`, no token needed for existence/gating metadata) —
  **both exist and are gated**. Confirmed with the user: target is
  `meta-llama/Llama-3.3-70B-Instruct`, since that's what the brief's SAE/NLA-coverage
  justification (layer 53, Neuronpedia tutorial pairing) is actually about, and it's
  the instruct-tuned variant refusal-behavior experiments need. `configs/models.yaml`
  already had this right; do not switch it to `Meta-Llama-3-70B`.
- `google/gemma-3-27b-it` live-checked the same way: exists, gated, and its HF
  `pipeline_tag` is `image-text-to-text` (Gemma 3 is multimodal) — noted in
  `scripts/download_models.py` since the plain `AutoModelForCausalLM` load pattern
  in section 3 above may need `Gemma3ForConditionalGeneration` or an equivalent
  text-only code path depending on the `transformers` version; verify at load time.
- **Not verified in this pass** (needs the actual libraries installed, or an
  HF_TOKEN for gated-content checks beyond existence): SAELens release strings
  (`gemma-scope-2-27b-it-res` etc.), the Llama 3.3 70B SAE release ID, and the
  Neuronpedia `/api/models` response (hit it directly — it serves the app shell,
  not a plain JSON endpoint from an unauthenticated curl; use the SDK/browser
  session once you have compute, not raw curl). `scripts/verify_repo_ids.py`
  implements section 0's checks precisely so this is a run-it-and-read-the-output
  job once `sae_lens` is installed, not a re-design job.
- Built to be run on Colab/HPC, not in this environment (no GPU, no HF_TOKEN
  configured here): `scripts/download_models.py`, `scripts/verify_repo_ids.py`,
  `scripts/sanity_check.py`, `src/refusal_mechinterp/models/backend.py`
  (`HFLocalBackend` now implemented against `transformer_lens`, previously a stub),
  `src/refusal_mechinterp/models/sae_loader.py`.
- `scripts/resolve_tier_c.py` — new, not in the original doc's script list but
  needed to actually resolve the UNRESOLVED tier-(c) placeholders left by
  `scripts/build_matched_triples.py` (see docs/DATASET_PLAN.md): uses
  `sentence-transformers` to rank seed-file candidates by semantic similarity to
  each placeholder's topic, for human review — it suggests, it does not
  auto-assign, since tier-(c) ground truth must stay human-verified per the
  dataset plan.
