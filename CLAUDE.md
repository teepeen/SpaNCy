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
- **Hybrid mode needs validation** — `mode="hybrid"` with `alpha=0.3` is the recommended default; tune alpha (0.1–0.5) based on kBET vs shape preservation.
- Training on 1.76M cells takes significant time. Use `--epochs 10` for testing, `--epochs 100` for production.
- Cross-batch contrastive loss: tune `w_cross_batch` (default 0.5) if kBET vs shape preservation tradeoff needs adjustment.
- GRL lambda ramps 0→1 over 30 epochs by default.
- kBET baseline (affine-only, 10 epochs): g1=0.88, g2=0.55, g3=0.19, g4=0.10, g5=0.53. Higher=better.
- Ensemble mode in `db_spancy_explore.ipynb` trains 3 models with diverse hyperparams; needs comparison with hybrid mode.

## kBET Metric Notes
- **Higher kBET acceptance rate = better** (batches well-mixed in local neighborhoods)
- **Lower chi-square = better**
- **Higher p-value = better**
- kBET operates in full 20D expression space — aligned 1D marginal histograms do NOT guarantee good kBET
- Computed on 5 clinical groups pairing samples from different batches
