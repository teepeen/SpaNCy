# SpaNCy — Project Context

## What This Is
**SpaNCy** (Spatial Neighborhood Contrastive CyCIF Normalizer) — Neural network-based normalization of CyCIF multiplexed imaging data. Targets the PRAD-CyCIF dataset (1.76M cells x 20 markers, 20 patients, 7 batches).

GitHub repo: https://github.com/teepeen/SpaNCy

## Architecture
```
X_raw + batch_id + (x,y) → CycleDegradationModel → SpatialGNNEncoder (GATv2) → Decoder + ProjectionHead + BatchDiscriminator
```
- **CycleDegradationModel**: batch(32d) + cycle(16d) embeddings → MLP → per-marker gamma/beta
- **SpatialGNNEncoder**: Linear→128 + 2x GATv2Conv(128, 4 heads) + residual + LayerNorm → 64d latent
- **Decoder**: 64→128→20, GELU + **Softplus** (not ReLU — avoids zero-spike artifact)
- **ProjectionHead**: 64→64→32, L2-normalized (training only)
- **BatchDiscriminator**: 64→32→n_batches with gradient reversal
- **Losses**: Huber (recon) + NT-Xent (contrastive, spatial-neighbor positives) + CE (adversarial)
- **Total**: `L_recon + 0.5*L_contrast + 0.3*L_adv`

## Files
| File | Purpose |
|------|---------|
| `spancy.py` | Full implementation (~850 lines): models, training, inference, CLI |
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
2. **Softplus instead of ReLU in decoder** — ReLU clips negative scaled values to exactly 0, creating a massive zero-spike after expm1. Softplus is smooth and strictly positive.
3. **AdjacencyIndex (CSR) for fast subgraph extraction** — Pre-built at training start. O(batch*k) per step instead of O(E) scanning all ~26M edges.
4. **Vectorized kNN graph building** — numpy broadcast instead of nested Python loops.
5. **cells_per_step default = 16000** — reduces steps/epoch from ~293 to ~110 for 1.76M cells.
6. **CosineAnnealingLR T_max floors at 1** — prevents ZeroDivisionError when n_epochs <= warmup_epochs.
7. **train() returns history dict** — tracks loss/recon/contrast/adv/lr/grl_lambda per epoch for plotting.

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
- **Zero-spike in normalized output** was caused by ReLU + StandardScaler combo. Fixed with Softplus + RobustScaler — needs re-verification after retraining.
- Training on 1.76M cells takes significant time even with optimizations. Consider reducing epochs for initial testing (`--epochs 10`).
- Contrastive loss uses spatial-neighbor positives only; cross-batch phenotype-matched positives (mentioned in plan) not yet implemented.
- GRL lambda ramps 0→1 over 30 epochs by default.
