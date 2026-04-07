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
- **Losses**: Huber (recon, penalizes delta magnitude) + NT-Xent (spatial contrastive) + CE (adversarial) + lower-quantile alignment (10th/25th percentiles only — safe for bimodal markers) + cross-batch NT-Xent (phenotype-matched cross-batch positives)
- **Total**: `L_recon + 0.5*L_contrast + 0.3*L_adv + 0.5*L_align + 0.5*L_cross_batch`

## Inference Pipeline (Three Modes)
1. **Affine correction** (`mode="affine"`): Only the CycleDegradationModel's learned gamma/beta are applied: `X_corrected = (X - beta) / gamma`. Pure shift+scale **per sample per marker** (20 correction groups) — perfectly preserves distribution shape but only corrects per-marker marginals (1D). kBET can be poor because inter-marker correlations are not corrected.
2. **Hybrid** (`mode="hybrid"`, default recommended): Affine correction + **scaled GNN residual delta**: `X_out = X_corrected + alpha * delta`. The `hybrid_alpha` parameter (default 0.3) controls the trade-off: 0=pure affine, 1=full residual. Gives multivariate alignment (improves kBET) while mostly preserving distribution shapes.
3. **Residual** (`mode="residual"`): Full pipeline with alpha=1.0. Best kBET but may distort distribution shapes.
4. **Per-sample mode alignment** (`align_samples=True`): Post-hoc step for any mode. For each marker, finds the histogram peak (mode) in each sample and shifts to match the global peak. Robust to bimodal markers (e.g. ECAD).
5. **Ensemble hybrid** (`normalize_adata_ensemble()`, `mode="hybrid"`): Averages gamma/beta AND GNN deltas across N ensemble models. Combines histogram stability of ensemble averaging with multivariate correction for kBET. Best of both worlds.

**Log-space cap**: Before expm1, values are clamped to 1.2x the raw data maximum in log1p space. Prevents astronomical output values from overcorrected gamma/beta.

## Files
| File | Purpose |
|------|---------|
| `spancy.py` | Full implementation (~1100 lines): models, training, inference, CLI |
| `requirements.txt` | Dependencies: torch, torch-geometric, anndata, numpy, scipy, scikit-learn |
| `spancy_explore.ipynb` | Dev/exploration notebook — step through pipeline interactively |
| `spancy_demo.ipynb` | Demo/tutorial — run SpaNCy, UMAP before/after, KS statistics |
| `db_spancy_explore.ipynb` | DBnorm-inspired notebook — per-marker R², ensemble training, single vs ensemble comparison, histogram PDF |

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
5. **Per-sample safe piecewise peak alignment in log space** — Detects peaks per marker per sample. Unimodal markers (or samples with only 1 detected peak): pure shift on leftmost peak. Bimodal markers where BOTH global and sample detect 2+ peaks: piecewise linear aligning both negative and positive peaks, with **clamped slope=1 beyond the positive peak** (pure shift in the tail). This corrects dynamic range differences between peaks without the tail-stretching artifact that occurs when the inter-peak scale factor is extrapolated.
6. **Lower-quantile alignment loss** — Matches only 10th/25th percentiles across samples (not full distribution). These quantiles are safely within the negative population for all CyCIF markers. Higher quantiles (50/75/90th) are excluded because they depend on biological mixture proportions. Replaced location_scale_loss (mean+var) which was also too coarse.
7. **Sample embeddings in CycleDegradationModel** — Per-sample (not just per-batch) affine correction: 20 correction groups instead of 7. Each marker uses its actual cycle embedding (not averaged). Eliminates need for post-hoc alignment in most cases.
8. **AdjacencyIndex (CSR) for fast subgraph extraction** — Pre-built at training start. O(batch*k) per step instead of O(E) scanning all ~26M edges.
9. **Vectorized kNN graph building** — numpy broadcast instead of nested Python loops.
10. **CosineAnnealingLR T_max floors at 1** — prevents ZeroDivisionError when n_epochs <= warmup_epochs.
11. **Explicit logger setup** — `StreamHandler(sys.stdout)` on the `spancy` logger, not `logging.basicConfig()`. Ensures output in Colab/Jupyter where root logger is pre-configured.
12. **train() returns history dict** — tracks loss/recon/contrast/adv/align/cross_batch/lr/grl_lambda per epoch.
13. **Cross-batch contrastive loss** — Spatial NT-Xent only pairs within-scene neighbors (same batch). Cross-batch NT-Xent finds phenotype-matched cells across batches (by cosine similarity on X_corrected) and uses them as positives on z_proj. Batch-balanced anchor sampling (512/step), top-5 cross-batch matches per anchor. Direct gradient to pull same-phenotype cells together across batches → improves kBET.
14. **Hybrid inference mode** — Affine-only gives good histograms (1D marginals) but poor kBET (20D multivariate). Full residual gives good kBET but distorts shapes. Hybrid `X_corrected + alpha*delta` (alpha=0.3) balances both. The key insight: per-marker affine correction can't fix inter-marker correlations that differ across batches.
15. **Log-space cap at inference** — Clamps normalized values in log space to 1.2x the raw data max before expm1. Prevents gamma/beta overcorrection from producing astronomical values (max went to 10^14 without this).
16. **DO NOT put alignment loss on X_corrected** — Tried 3 times with different regularization strategies; all caused gamma/beta explosion because recon loss `huber(X_recon, X_corrected)` has no anchor to original data. Keep alignment on `X_recon` only (indirect gradient through GNN).
17. **DBnorm-inspired ensemble** — Train N models with diverse hyperparameters (varying LR, loss weights, k-neighbors), average gamma/beta at inference. Inspired by BER-Bagging (Bararpour et al., Sci Rep 2021). Stabilizes corrections.
18. **Per-marker batch adj-R²** — Regress each marker on batch labels (one-hot). Simple diagnostic: high adj-R² after correction = that marker still has residual batch effect.
19. **Ensemble hybrid inference** (`normalize_adata_ensemble()`) — Each model computes its own full correction path (own gamma/beta → own GNN delta), then final outputs are averaged: `mean_i(X_corrected_i + alpha * delta_i)`. Critical: each GNN must see the X_corrected it was trained on (not an ensemble-averaged X_corrected) for strong deltas. First attempt with shared X_corrected gave weaker results due to distribution mismatch.

## Colab Usage
Both notebooks have Section 0 (Colab Setup) that:
1. Installs torch-geometric by auto-detecting Colab's torch+CUDA version
2. Uploads `spancy.py` via `google.colab.files.upload()` (lands in `/content/`, already on sys.path)
3. **After re-uploading spancy.py**: must `importlib.reload(spancy)` or restart runtime

## CLI Usage
```bash
python spancy.py --input PRAD_anndata.h5ad --output PRAD_normalized.h5ad --epochs 100 --device cuda
```

## Results So Far (2026-04-07)

### Batch adj-R² (lower = better, less batch effect)
| Method | Mean adj-R² | Notes |
|--------|-------------|-------|
| Raw | 0.259 | Baseline |
| Single model (affine) | 0.247 | Barely improves; some markers WORSE (ChromA 0.10→0.41, CD20 0.47→0.68) |
| Ensemble 3x (affine) | **0.044** | 83% reduction. Nearly every marker below 0.05. |

Ensemble outliers: Ki67 (0.13), NOTCH1 (0.24), FOXA1 (0.10) still have residual batch effect.

### kBET (higher = better, batches well-mixed) — first ensemble hybrid run (shared X_corrected, now fixed)
| Method | Mean kBET | Mean chi² |
|--------|-----------|-----------|
| Single model (affine) | 0.176 | 24.9 |
| Ensemble affine | 0.418 | 15.8 |
| Ensemble hybrid a0.2 | 0.574 | 13.9 |

**Pending**: Re-running kBET with per-model correction path fix (each GNN sees its own X_corrected). Results expected to improve further.

## Known Issues / Next Steps
- **Ensemble hybrid kBET re-run pending** — per-model correction paths should give stronger GNN deltas. Testing alpha range 0.1–1.0.
- Ki67, NOTCH1, FOXA1 have residual batch effect even with ensemble — may need more models (5 instead of 3) or more epochs.
- g5 group sometimes returns NaN in kBET — numerical instability in pegasus, now handled with NaN detection.
- Training on 1.76M cells takes significant time. Use `--epochs 10` for testing, `--epochs 100` for production.
- GRL lambda ramps 0→1 over 30 epochs by default.

## kBET Metric Notes
- **Higher kBET acceptance rate = better** (batches well-mixed in local neighborhoods)
- **Lower chi-square = better**
- **Higher p-value = better**
- kBET operates in full 20D expression space — aligned 1D marginal histograms do NOT guarantee good kBET
- Computed on 5 clinical groups pairing samples from different batches (rep="umap")
