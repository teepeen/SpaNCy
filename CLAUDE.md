# SpaNCy — Project Context

## What This Is
**SpaNCy** (Spatial Neighborhood Contrastive CyCIF Normalizer) — Neural network-based normalization of CyCIF multiplexed imaging data. Targets the PRAD-CyCIF dataset (1.76M cells x 20 markers, 20 patients, 7 batches).

GitHub repo: https://github.com/teepeen/SpaNCy

## Architecture
```
X_raw + batch_id + (x,y) → CycleDegradationModel → SpatialGNNEncoder (GATv2) → ResidualDecoder + ProjectionHead + BatchDiscriminator
```
- **CycleDegradationModel**: batch(32d) + sample(16d) + per-marker-cycle(16d) embeddings → shared MLP(64→64→2) → per-marker gamma/beta. MLP is applied per marker with that marker's actual cycle embedding (not averaged).
- **SpatialGNNEncoder**: Linear→128 + 2x GATv2Conv(128, 4 heads) + residual + LayerNorm → 64d latent
- **ResidualDecoder**: 64→128→20, GELU, no output activation. Outputs a **delta** (correction), not the full expression. Final output = `X_corrected + delta`. Initialized near-zero so training starts from identity.
- **ProjectionHead**: 64→64→32, L2-normalized (training only)
- **BatchDiscriminator**: 64→32→n_batches with gradient reversal
- **Losses**: Huber (recon, penalizes delta magnitude) + NT-Xent (contrastive) + CE (adversarial) + lower-quantile alignment (10th/25th percentiles only — safe for bimodal markers)
- **Total**: `L_recon + 0.5*L_contrast + 0.3*L_adv + 0.5*L_align`

## Inference Pipeline (Two-Stage)
At inference, the GNN/decoder are **not used** — they serve only as a training scaffold.
1. **Affine correction** (default `mode="affine"`): Only the CycleDegradationModel's learned gamma/beta are applied: `X_corrected = (X - beta) / gamma`. This is a pure shift+scale **per sample per marker** (20 correction groups, not just 7 batches) — perfectly preserves distribution shape.
2. **Per-sample mode alignment** (`align_samples=True`): For each marker, finds the histogram peak (mode) in each sample and shifts to match the global peak. Unlike median alignment, this is robust to bimodal markers (e.g. ECAD) where the median depends on positive cell fraction (biology). Default `False` since the model's per-sample affine correction usually handles this.

Alternative: `mode="residual"` uses the full GNN pipeline (better kBET but distorts distribution shape).

## Files
| File | Purpose |
|------|---------|
| `spancy.py` | Full implementation (~1050 lines): models, training, inference, CLI |
| `requirements.txt` | Dependencies: torch, torch-geometric, anndata, numpy, scipy, scikit-learn |
| `spancy_explore.ipynb` | Dev/exploration notebook — step through pipeline interactively |
| `spancy_demo.ipynb` | Demo/tutorial — run SpaNCy, UMAP before/after, KS statistics |

## Dataset-Specific Details (PRAD-CyCIF)
- **obs columns**: `cell_id`, `sample_id`, `scene_id`, `batch_id`, `x`, `y`
- **var**: `marker_name` column holds actual names (var_names are numeric 0-19 by default)
- `load_adata()` auto-detects `batch_id`→`batch`, `scene_id` for graphs, and sets `var_names` from `marker_name`
- Default cycle config maps 20 markers to 6 imaging cycles (configurable via `--cycle_config` JSON)

## Key Design Decisions & Fixes Applied
1. **RobustScaler instead of StandardScaler** — CyCIF data is zero-inflated; mean/std scaling biases everything toward zero. RobustScaler (median/IQR) handles this correctly.
2. **No output activation in decoder** — ReLU caused zero-spike, Softplus caused left-edge compression (can't represent negative scaled values). Decoder now outputs unconstrained values in scaled space; non-negativity enforced after full inverse transform.
3. **ResidualDecoder instead of full Decoder** — Full encode→decode through GNN bottleneck distorts distribution shapes (smooths bimodality). Residual approach: `output = X_corrected + small_delta` preserves original signal, model only learns corrections.
4. **Affine-only inference** — GNN/decoder used only during training as scaffold. At inference, only CycleDegradationModel correction applied (shape-preserving).
5. **Per-sample mode alignment in log space** — Uses histogram peak (mode) instead of median. Robust to bimodal markers (ECAD, CD45) where median depends on positive cell fraction (biology). Done in log space (before expm1) so shifts are multiplicative — no zero-clipping.
6. **Lower-quantile alignment loss** — Matches only 10th/25th percentiles across samples (not full distribution). These quantiles are safely within the negative population for all CyCIF markers. Higher quantiles (50/75/90th) are excluded because they depend on biological mixture proportions. Replaced location_scale_loss (mean+var) which was also too coarse.
7. **Sample embeddings in CycleDegradationModel** — Per-sample (not just per-batch) affine correction: 20 correction groups instead of 7. Each marker uses its actual cycle embedding (not averaged). Eliminates need for post-hoc alignment in most cases.
8. **AdjacencyIndex (CSR) for fast subgraph extraction** — Pre-built at training start. O(batch*k) per step instead of O(E) scanning all ~26M edges.
9. **Vectorized kNN graph building** — numpy broadcast instead of nested Python loops.
10. **CosineAnnealingLR T_max floors at 1** — prevents ZeroDivisionError when n_epochs <= warmup_epochs.
11. **Explicit logger setup** — `StreamHandler(sys.stdout)` on the `spancy` logger, not `logging.basicConfig()`. Ensures output in Colab/Jupyter where root logger is pre-configured.
12. **train() returns history dict** — tracks loss/recon/contrast/adv/align/lr/grl_lambda per epoch.

## Colab Usage
Both notebooks have Section 0 (Colab Setup) that:
1. Installs torch-geometric by auto-detecting Colab's torch+CUDA version
2. Uploads `spancy.py` via `google.colab.files.upload()` (lands in `/content/`, already on sys.path)
3. **After re-uploading spancy.py**: must `importlib.reload(spancy)` or restart runtime

## CLI Usage
```bash
python spancy.py --input PRAD_anndata.h5ad --output PRAD_normalized.h5ad --epochs 100 --device cuda
```

## Known Issues / Next Steps
- Per-sample affine correction + lower-quantile alignment should handle most batch effects. Mode alignment post-hoc available for additional refinement.
- Training on 1.76M cells takes significant time even with optimizations. Consider reducing epochs for initial testing (`--epochs 10`).
- Contrastive loss uses spatial-neighbor positives only; cross-batch phenotype-matched positives not yet implemented.
- GRL lambda ramps 0→1 over 30 epochs by default.
- Need to validate with kBET and positive population preservation metrics side-by-side.
