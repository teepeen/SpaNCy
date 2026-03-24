#!/usr/bin/env python
"""
SpaNCy: Spatial Neighborhood Contrastive CyCIF Normalizer

Neural network-based normalization of CyCIF multiplexed imaging data using
spatial graph attention networks, cycle-aware artifact modeling, and
self-supervised contrastive learning.
"""

import argparse
import json
import logging
import math
import sys
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv

log = logging.getLogger("spancy")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_handler)
    log.propagate = False  # avoid duplicate messages if root logger is also configured

# ──────────────────────────────────────────────────────────────────────────────
# Default cycle assignment for PRAD-CyCIF 20-marker panel
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CYCLE_CONFIG: Dict[int, List[str]] = {
    0: ["DAPI", "DAPI_R1"],  # accept either naming convention
    1: ["EPCAM", "CD56", "CD45"],
    2: ["aSMA", "ChromA", "CK14", "Ki67"],
    3: ["GZMB", "ECAD", "PD1", "CD31"],
    4: ["CD45RA", "HLADRB1", "CD3", "p53"],
    5: ["FOXA1", "CDX2", "CD20", "NOTCH1"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────


def load_adata(path: str) -> ad.AnnData:
    """Load an AnnData object and validate expected fields."""
    log.info("Loading %s", path)
    adata = ad.read_h5ad(path)
    log.info("Loaded %d cells x %d markers", adata.n_obs, adata.n_vars)

    # Detect batch column (supports "batch", "batch_id", "Batch", etc.)
    batch_col = None
    for col in ("batch", "batch_id", "Batch", "BatchID", "batch_ID"):
        if col in adata.obs.columns:
            batch_col = col
            break
    if batch_col is None:
        raise ValueError(
            "adata.obs must contain a batch column "
            "(looked for: batch, batch_id, Batch, BatchID)"
        )
    # Standardize to "batch" for downstream use
    if batch_col != "batch":
        log.info("Using '%s' as batch column", batch_col)
        adata.obs["batch"] = adata.obs[batch_col]

    # If var_names are numeric indices, look for a marker_name column in .var
    if all(v.isdigit() for v in adata.var_names.astype(str)):
        for col in ("marker_name", "marker", "Marker", "gene", "Gene", "protein"):
            if col in adata.var.columns:
                log.info("Setting var_names from var['%s']", col)
                adata.var_names = adata.var[col].values
                break
        else:
            log.warning(
                "var_names are numeric and no marker_name column found in .var; "
                "cycle assignment may not work correctly"
            )

    # Ensure spatial coordinates exist
    has_spatial = "spatial" in adata.obsm
    has_xy = {"x", "y"}.issubset(adata.obs.columns)
    if not has_spatial and not has_xy:
        raise ValueError(
            "Need spatial coordinates: adata.obsm['spatial'] or adata.obs[['x','y']]"
        )
    return adata


def get_spatial_coords(adata: ad.AnnData) -> np.ndarray:
    """Extract spatial (x, y) coordinates as an (N, 2) array."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])[:, :2]
    else:
        coords = adata.obs[["x", "y"]].values
    return coords.astype(np.float64)


def assign_marker_cycles(
    marker_names: List[str],
    cycle_config: Dict[int, List[str]],
) -> np.ndarray:
    """Return an array mapping each marker index to its imaging cycle."""
    marker_to_cycle = {}
    for cyc, markers in cycle_config.items():
        for m in markers:
            marker_to_cycle[m] = int(cyc)

    cycles = np.zeros(len(marker_names), dtype=np.int64)
    for i, name in enumerate(marker_names):
        if name in marker_to_cycle:
            cycles[i] = marker_to_cycle[name]
        else:
            log.warning("Marker '%s' not found in cycle config; defaulting to cycle 0", name)
    return cycles


def log1p_scale(
    X: np.ndarray,
) -> Tuple[np.ndarray, RobustScaler]:
    """Apply log1p then per-marker RobustScaler (median/IQR).

    RobustScaler is much less sensitive to the zero-inflated distributions
    typical of CyCIF imaging data than StandardScaler.
    """
    X_log = np.log1p(np.clip(X, 0, None))
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_log)
    return X_scaled.astype(np.float32), scaler


def build_knn_graph(
    coords: np.ndarray,
    scene_ids: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """Build k-NN spatial edges per scene. Returns (2, E) edge index array."""
    scenes = np.unique(scene_ids)
    edge_parts = []

    for sc in scenes:
        mask = scene_ids == sc
        idx = np.where(mask)[0]
        n_sc = len(idx)
        k_sc = min(k, n_sc - 1)
        if k_sc < 1:
            continue
        tree = cKDTree(coords[idx])
        _, neighbors = tree.query(coords[idx], k=k_sc + 1)
        neighbors = neighbors[:, 1:]  # drop self

        # Vectorized: build (n_sc * k_sc) edges at once
        src_local = np.repeat(np.arange(n_sc), k_sc)
        dst_local = neighbors.ravel()
        src_global = idx[src_local]
        dst_global = idx[dst_local]
        edge_parts.append(np.stack([src_global, dst_global], axis=0))

    if not edge_parts:
        return np.zeros((2, 0), dtype=np.int64)
    return np.concatenate(edge_parts, axis=1).astype(np.int64)


def get_scene_ids(adata: ad.AnnData) -> np.ndarray:
    """Determine per-cell scene IDs for graph construction."""
    for col in ("scene_id", "scene", "Scene", "SCENE", "sample_id", "sample", "Sample", "image_id"):
        if col in adata.obs.columns:
            log.info("Using '%s' as scene column for spatial graph", col)
            return adata.obs[col].astype("category").cat.codes.values
    # Fall back to batch as scene proxy
    log.warning("No scene column found; using 'batch' for spatial graph construction")
    return adata.obs["batch"].astype("category").cat.codes.values


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ──────────────────────────────────────────────────────────────────────────────


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


# ──────────────────────────────────────────────────────────────────────────────
# Model Components
# ──────────────────────────────────────────────────────────────────────────────


class CycleDegradationModel(nn.Module):
    """Learns per-batch, per-sample, per-cycle scale (gamma) and shift (beta).

    Each marker's correction is conditioned on:
    - batch identity (32d embedding) — captures batch-level systematic effects
    - sample identity (16d embedding) — captures within-batch sample variation
    - marker's imaging cycle (16d embedding) — captures cycle-specific degradation

    The MLP is shared across markers but receives per-marker cycle embeddings,
    so markers in different imaging cycles get different corrections.
    """

    def __init__(self, n_batches: int, n_samples: int, n_cycles: int, n_markers: int):
        super().__init__()
        self.batch_embed = nn.Embedding(n_batches, 32)
        self.sample_embed = nn.Embedding(n_samples, 16)
        self.cycle_embed = nn.Embedding(n_cycles, 16)
        self.n_markers = n_markers

        # Shared MLP applied per marker: (batch_32 + sample_16 + cycle_16) = 64 → 2
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_ids: (B,) batch index per cell
            sample_ids: (B,) sample index per cell
            marker_cycles: (M,) cycle index per marker
        Returns:
            gamma: (B, M) scale (strictly positive via softplus)
            beta:  (B, M) shift
        """
        b_emb = self.batch_embed(batch_ids)      # (B, 32)
        s_emb = self.sample_embed(sample_ids)    # (B, 16)
        c_emb = self.cycle_embed(marker_cycles)  # (M, 16)

        # Build per-cell, per-marker input: (B, M, 64)
        B = b_emb.size(0)
        M = self.n_markers
        bs = torch.cat([b_emb, s_emb], dim=-1)    # (B, 48)
        bs = bs.unsqueeze(1).expand(B, M, 48)     # (B, M, 48)
        c = c_emb.unsqueeze(0).expand(B, M, 16)   # (B, M, 16)
        h = torch.cat([bs, c], dim=-1)             # (B, M, 64)

        out = self.mlp(h)                          # (B, M, 2)
        gamma = F.softplus(out[..., 0]) + 0.1     # (B, M)
        beta = out[..., 1]                         # (B, M)
        return gamma, beta

    def correct(
        self,
        X: torch.Tensor,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self(batch_ids, sample_ids, marker_cycles)
        return (X - beta) / gamma


class SpatialGNNEncoder(nn.Module):
    """Two-layer GATv2 encoder producing spatial-aware latent embeddings."""

    def __init__(self, n_markers: int, hidden: int = 128, latent: int = 64, heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(n_markers, hidden)
        self.gat1 = GATv2Conv(hidden, hidden // heads, heads=heads, concat=True)
        self.gat2 = GATv2Conv(hidden, hidden // heads, heads=heads, concat=True)
        self.out_proj = nn.Linear(hidden, latent)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.norm1(F.gelu(self.gat1(h, edge_index)) + h)
        h = self.norm2(F.gelu(self.gat2(h, edge_index)) + h)
        return self.out_proj(h)


class ResidualDecoder(nn.Module):
    """Predicts a small per-cell correction (delta) from the latent.

    Output = X_corrected + delta.  The Huber reconstruction loss keeps delta
    small (preserving original distribution shape), while the alignment and
    adversarial losses provide gradient for batch-specific adjustments.

    Initialized with near-zero final layer so delta ≈ 0 at the start of
    training (identity-like initialization).
    """

    def __init__(self, latent: int = 64, hidden: int = 128, n_markers: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_markers),
        )
        # Initialize final linear layer near zero so initial delta ≈ 0
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns the residual correction (delta), NOT the full output."""
        return self.net(z)


class ProjectionHead(nn.Module):
    """Projects latent to contrastive embedding space (training only)."""

    def __init__(self, latent: int = 64, proj: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, proj),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = self.net(z)
        return F.normalize(p, dim=-1)


class BatchDiscriminator(nn.Module):
    """Predicts batch from latent (with gradient reversal for adversarial training)."""

    def __init__(self, latent: int = 64, n_batches: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 32),
            nn.GELU(),
            nn.Linear(32, n_batches),
        )

    def forward(self, z: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        z_rev = grad_reverse(z, grl_lambda)
        return self.net(z_rev)


class SpaNCy(nn.Module):
    """Combined SpaNCy model."""

    def __init__(
        self,
        n_markers: int,
        n_batches: int,
        n_samples: int,
        n_cycles: int,
        hidden: int = 128,
        latent: int = 64,
        heads: int = 4,
        proj_dim: int = 32,
    ):
        super().__init__()
        self.cycle_model = CycleDegradationModel(n_batches, n_samples, n_cycles, n_markers)
        self.encoder = SpatialGNNEncoder(n_markers, hidden, latent, heads)
        self.decoder = ResidualDecoder(latent, hidden, n_markers)
        self.proj_head = ProjectionHead(latent, proj_dim)
        self.batch_disc = BatchDiscriminator(latent, n_batches)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        X_corrected = self.cycle_model.correct(X, batch_ids, sample_ids, marker_cycles)
        z = self.encoder(X_corrected, edge_index)
        delta = self.decoder(z)
        X_recon = X_corrected + delta  # residual correction
        z_proj = self.proj_head(z)
        batch_logits = self.batch_disc(z, grl_lambda)

        return {
            "X_corrected": X_corrected,
            "z": z,
            "delta": delta,
            "X_recon": X_recon,
            "z_proj": z_proj,
            "batch_logits": batch_logits,
        }

    @torch.no_grad()
    def normalize(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        batch_ids: torch.Tensor,
        sample_ids: torch.Tensor,
        marker_cycles: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        X_corrected = self.cycle_model.correct(X, batch_ids, sample_ids, marker_cycles)
        z = self.encoder(X_corrected, edge_index)
        delta = self.decoder(z)
        return X_corrected + delta


# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────


def nt_xent_loss(
    z_proj: torch.Tensor,
    edge_index: torch.Tensor,
    tau: float = 0.1,
    max_positives: int = 4096,
) -> torch.Tensor:
    """NT-Xent with spatial-neighbor positives.

    For each cell i, its positive set is the set of cells connected to it
    in the spatial k-NN graph. We sample a subset of edges to keep memory bounded.
    """
    n = z_proj.size(0)
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=z_proj.device)

    # Subsample edges if too many
    n_edges = edge_index.size(1)
    if n_edges > max_positives:
        perm = torch.randperm(n_edges, device=edge_index.device)[:max_positives]
        edge_index = edge_index[:, perm]

    src, dst = edge_index[0], edge_index[1]

    # Pairwise similarity for anchors involved in positive pairs
    unique_nodes = torch.unique(torch.cat([src, dst]))
    if unique_nodes.size(0) < 2:
        return torch.tensor(0.0, device=z_proj.device)

    # Build local index mapping
    node_map = torch.full((n,), -1, dtype=torch.long, device=z_proj.device)
    node_map[unique_nodes] = torch.arange(unique_nodes.size(0), device=z_proj.device)

    z_sub = z_proj[unique_nodes]  # (U, D)
    sim = torch.mm(z_sub, z_sub.t()) / tau  # (U, U)

    # Mask out self-similarity
    mask_self = torch.eye(z_sub.size(0), device=z_proj.device, dtype=torch.bool)
    sim.masked_fill_(mask_self, -1e9)

    src_local = node_map[src]
    dst_local = node_map[dst]

    # For each positive pair (src, dst), the loss is:
    #   -log( exp(sim(src,dst)) / sum_j!=src exp(sim(src,j)) )
    pos_sim = sim[src_local, dst_local]  # (E,)
    log_denom = torch.logsumexp(sim[src_local], dim=-1)  # (E,)
    loss = (-pos_sim + log_denom).mean()
    return loss


def cross_batch_nt_xent_loss(
    z_proj: torch.Tensor,
    X_corrected: torch.Tensor,
    batch_ids: torch.Tensor,
    tau: float = 0.1,
    n_anchors: int = 512,
    k_positives: int = 5,
) -> torch.Tensor:
    """NT-Xent with phenotype-matched cross-batch positives.

    For each anchor cell, finds cells from OTHER batches with the most
    similar expression profiles (cosine similarity on X_corrected) and
    uses them as positives in the contrastive loss computed on z_proj.

    This gives direct gradient to pull together cells that share biology
    but come from different batches — the key signal missing from the
    spatial-only NT-Xent (which only pairs within-scene neighbors).
    """
    n = z_proj.size(0)
    unique_batches = torch.unique(batch_ids)
    if unique_batches.size(0) < 2 or n < 10:
        return torch.tensor(0.0, device=z_proj.device)

    # Sample anchors (balanced across batches)
    anchor_indices = []
    per_batch_budget = max(1, n_anchors // unique_batches.size(0))
    for b in unique_batches:
        b_mask = (batch_ids == b).nonzero(as_tuple=True)[0]
        if b_mask.size(0) == 0:
            continue
        n_take = min(per_batch_budget, b_mask.size(0))
        perm = torch.randperm(b_mask.size(0), device=z_proj.device)[:n_take]
        anchor_indices.append(b_mask[perm])
    if not anchor_indices:
        return torch.tensor(0.0, device=z_proj.device)
    anchors = torch.cat(anchor_indices)  # (A,)
    n_anchors_actual = anchors.size(0)

    # Expression similarity to find cross-batch phenotype matches
    X_norm = F.normalize(X_corrected, dim=-1)       # (N, M)
    anchor_X = X_norm[anchors]                       # (A, M)
    expr_sim = torch.mm(anchor_X, X_norm.t())        # (A, N)

    # Mask same-batch cells (cannot be cross-batch positives)
    anchor_batches = batch_ids[anchors]               # (A,)
    same_batch = anchor_batches.unsqueeze(1) == batch_ids.unsqueeze(0)  # (A, N)
    expr_sim[same_batch] = -1e9

    # Check that each anchor has at least one cross-batch candidate
    has_candidates = (expr_sim > -1e8).any(dim=1)
    if not has_candidates.any():
        return torch.tensor(0.0, device=z_proj.device)
    valid = has_candidates.nonzero(as_tuple=True)[0]
    anchors = anchors[valid]
    expr_sim = expr_sim[valid]
    n_anchors_actual = anchors.size(0)

    # Top-k cross-batch phenotype matches per anchor
    n_cross = (expr_sim > -1e8).sum(dim=1).min().item()
    k = min(k_positives, max(1, int(n_cross)))
    _, topk_idx = expr_sim.topk(k, dim=1)            # (A, k)

    # Contrastive loss on z_proj (already L2-normalized)
    z_anchor = z_proj[anchors]                        # (A, D)
    z_sim = torch.mm(z_anchor, z_proj.t()) / tau      # (A, N)

    # Mask self from denominator
    self_mask = torch.zeros(n_anchors_actual, n, dtype=torch.bool, device=z_proj.device)
    self_mask[torch.arange(n_anchors_actual, device=z_proj.device), anchors] = True
    z_sim[self_mask] = -1e9

    log_denom = torch.logsumexp(z_sim, dim=-1)       # (A,)
    pos_sim = z_sim.gather(1, topk_idx)               # (A, k)
    loss = (-pos_sim + log_denom.unsqueeze(1)).mean()

    return loss


def quantile_alignment_loss(
    X_recon: torch.Tensor,
    sample_ids: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.1, 0.25),
) -> torch.Tensor:
    """Lower-quantile alignment across samples.

    Only matches the 10th and 25th percentiles of each marker across samples.
    These quantiles fall safely within the **negative (background) population**
    for CyCIF markers, regardless of the positive cell fraction.

    Higher quantiles (50th, 75th, 90th) are deliberately excluded because for
    bimodal markers (e.g. ECAD, CD45) they depend on the biological mixture
    proportion (% positive cells), which varies across samples.  Forcing those
    to match distorts biology — some samples get over-corrected.

    Two lower quantiles are sufficient to constrain both location (where the
    negative peak sits) and scale (how wide it is) of the background population.
    """
    unique_samples = torch.unique(sample_ids)
    if unique_samples.size(0) < 2:
        return torch.tensor(0.0, device=X_recon.device)

    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=X_recon.device)

    # Global quantiles across all cells: (Q, M)
    global_q = torch.quantile(X_recon, q_tensor, dim=0)
    # Scale factor: spread between the two quantiles, per marker
    q_spread = (global_q[-1] - global_q[0]).clamp(min=1e-6)  # (M,)

    loss = torch.tensor(0.0, device=X_recon.device)
    n_counted = 0

    for s in unique_samples:
        mask = sample_ids == s
        if mask.sum() < 50:
            continue
        sample_q = torch.quantile(X_recon[mask], q_tensor, dim=0)  # (Q, M)
        # Normalized squared error on lower quantiles only
        loss = loss + (((sample_q - global_q) / q_spread) ** 2).mean()
        n_counted += 1

    if n_counted == 0:
        return torch.tensor(0.0, device=X_recon.device)

    return loss / n_counted


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────


class SpatialClusterSampler:
    """Mini-batch sampler using spatial k-means clusters across batches."""

    def __init__(
        self,
        coords: np.ndarray,
        batch_ids: np.ndarray,
        cluster_size: int = 500,
        cells_per_step: int = 6000,
        seed: int = 42,
    ):
        self.coords = coords
        self.batch_ids = batch_ids
        self.cluster_size = cluster_size
        self.cells_per_step = cells_per_step
        self.rng = np.random.RandomState(seed)

        n_clusters = max(1, len(coords) // cluster_size)
        log.info("Fitting %d spatial clusters...", n_clusters)
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=4096)
        self.cluster_labels = km.fit_predict(coords)
        self.n_clusters = n_clusters
        self.unique_batches = np.unique(batch_ids)

    def sample(self) -> np.ndarray:
        """Return indices for one mini-batch."""
        indices = []
        target_per_batch = self.cells_per_step // len(self.unique_batches)

        for b in self.unique_batches:
            b_mask = self.batch_ids == b
            b_indices = np.where(b_mask)[0]
            b_clusters = self.cluster_labels[b_mask]
            unique_cl = np.unique(b_clusters)

            # Sample clusters, gather their cells
            n_cl_sample = max(1, target_per_batch // self.cluster_size)
            chosen_cl = self.rng.choice(unique_cl, size=min(n_cl_sample, len(unique_cl)), replace=False)

            for cl in chosen_cl:
                cl_idx = b_indices[b_clusters == cl]
                if len(cl_idx) > self.cluster_size:
                    cl_idx = self.rng.choice(cl_idx, size=self.cluster_size, replace=False)
                indices.append(cl_idx)

        return np.concatenate(indices) if indices else np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return max(1, len(self.coords) // self.cells_per_step)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────


class AdjacencyIndex:
    """Pre-built CSR adjacency for fast subgraph extraction.

    Instead of scanning all E edges every step, we gather only edges
    for the mini-batch nodes. O(batch_size * k) instead of O(E).
    """

    def __init__(self, edge_index: np.ndarray, n_nodes: int):
        src, dst = edge_index[0], edge_index[1]
        # Store as CSR: indptr + destinations
        order = np.argsort(src)
        self._dst = dst[order].copy()
        self._src_sorted = src[order]
        self._indptr = np.searchsorted(self._src_sorted, np.arange(n_nodes + 1))
        self._n_nodes = n_nodes

    def subgraph(self, node_indices: np.ndarray) -> Tuple[torch.Tensor, int]:
        """Extract and re-index edges for a subset of nodes (vectorized)."""
        n_local = len(node_indices)
        if n_local == 0:
            return torch.zeros((2, 0), dtype=torch.long), 0

        # Gather all outgoing edges from batch nodes
        # Build src_repeat and dst_all by slicing CSR
        src_parts = []
        dst_parts = []
        for local_i, g in enumerate(node_indices):
            s, e = self._indptr[g], self._indptr[g + 1]
            if s < e:
                dst_parts.append(self._dst[s:e])
                src_parts.append(np.full(e - s, local_i, dtype=np.int64))

        if not src_parts:
            return torch.zeros((2, 0), dtype=torch.long), n_local

        src_all = np.concatenate(src_parts)
        dst_global = np.concatenate(dst_parts)

        # Filter: keep only edges where dst is also in batch
        in_batch = np.zeros(self._n_nodes, dtype=np.bool_)
        in_batch[node_indices] = True
        mask = in_batch[dst_global]

        if not mask.any():
            return torch.zeros((2, 0), dtype=torch.long), n_local

        # Re-index dst to local
        g2l = np.empty(self._n_nodes, dtype=np.int64)
        g2l[node_indices] = np.arange(n_local, dtype=np.int64)

        src_local = src_all[mask]
        dst_local = g2l[dst_global[mask]]
        edge_local = torch.from_numpy(np.stack([src_local, dst_local], axis=0))
        return edge_local, n_local


def build_subgraph(
    edge_index: np.ndarray,
    node_indices: np.ndarray,
) -> Tuple[torch.Tensor, int]:
    """Extract subgraph for a set of nodes and re-index edges (vectorized)."""
    n_local = len(node_indices)
    if n_local == 0:
        return torch.zeros((2, 0), dtype=torch.long), 0

    # Vectorized membership test using a boolean array
    max_id = max(edge_index.max(), node_indices.max()) + 1
    in_batch = np.zeros(max_id, dtype=np.bool_)
    in_batch[node_indices] = True

    src, dst = edge_index[0], edge_index[1]
    mask = in_batch[src] & in_batch[dst]

    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long), n_local

    # Re-index with vectorized lookup
    g2l = np.empty(max_id, dtype=np.int64)
    g2l[node_indices] = np.arange(n_local, dtype=np.int64)

    sub_src = g2l[src[mask]]
    sub_dst = g2l[dst[mask]]
    edge_local = torch.from_numpy(np.stack([sub_src, sub_dst], axis=0))
    return edge_local, n_local


def train(
    adata: ad.AnnData,
    cycle_config: Dict[int, List[str]],
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_neighbors: int = 15,
    cluster_size: int = 500,
    cells_per_step: int = 12000,
    device_str: str = "cpu",
    warmup_epochs: int = 5,
    grl_ramp_epochs: int = 30,
    w_contrast: float = 0.5,
    w_adv: float = 0.3,
    w_align: float = 0.5,
    w_cross_batch: float = 0.5,
    tau: float = 0.1,
    cross_batch_anchors: int = 512,
    cross_batch_k: int = 5,
) -> Tuple["SpaNCy", RobustScaler, np.ndarray, Dict[str, List[float]]]:
    """Full training loop. Returns (model, scaler, marker_cycles, history).

    history keys: 'loss', 'recon', 'contrast', 'adv', 'align', 'cross_batch',
    'lr', 'grl_lambda' — one entry per epoch (averaged over steps).
    """
    device = torch.device(device_str)

    # -- Preprocessing --
    X_raw = np.asarray(adata.X)
    if sp.issparse(adata.X):
        X_raw = adata.X.toarray()
    X_scaled, scaler = log1p_scale(X_raw)

    marker_names = list(adata.var_names)
    marker_cycles = assign_marker_cycles(marker_names, cycle_config)
    marker_cycles_t = torch.tensor(marker_cycles, dtype=torch.long, device=device)
    n_markers = len(marker_names)
    n_cycles = int(marker_cycles.max()) + 1

    # Encode batch labels
    batch_cats = adata.obs["batch"].astype("category")
    batch_codes = batch_cats.cat.codes.values.astype(np.int64)
    n_batches = int(batch_codes.max()) + 1

    # Encode sample labels (finer-grained than batch)
    sample_col = None
    for col in ("sample_id", "sample", "Sample", "patient_id", "patient"):
        if col in adata.obs.columns:
            sample_col = col
            break
    if sample_col is None:
        sample_col = "batch"
        log.warning("No sample column found; using 'batch' as sample proxy")
    sample_cats = adata.obs[sample_col].astype("category")
    sample_codes = sample_cats.cat.codes.values.astype(np.int64)
    n_samples = int(sample_codes.max()) + 1
    log.info(
        "Detected %d batches, %d samples (col='%s'), %d cycles, %d markers",
        n_batches, n_samples, sample_col, n_cycles, n_markers,
    )

    # Spatial graph
    coords = get_spatial_coords(adata)
    scene_ids = get_scene_ids(adata)
    log.info("Building k-NN spatial graph (k=%d)...", k_neighbors)
    edge_index_np = build_knn_graph(coords, scene_ids, k=k_neighbors)
    log.info("Spatial graph: %d edges", edge_index_np.shape[1])

    # Pre-build adjacency index for fast subgraph extraction
    log.info("Building adjacency index...")
    adj_index = AdjacencyIndex(edge_index_np, adata.n_obs)

    # -- Model --
    model = SpaNCy(
        n_markers=n_markers,
        n_batches=n_batches,
        n_samples=n_samples,
        n_cycles=n_cycles,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", param_count)

    # -- Optimizer + Scheduler --
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # -- Sampler --
    sampler = SpatialClusterSampler(
        coords, batch_codes, cluster_size=cluster_size, cells_per_step=cells_per_step
    )
    steps_per_epoch = len(sampler)

    # -- Tensors --
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    batch_all = torch.tensor(batch_codes, dtype=torch.long)
    sample_all = torch.tensor(sample_codes, dtype=torch.long)

    # -- Training loop --
    huber = nn.HuberLoss(delta=1.0)
    ce = nn.CrossEntropyLoss()

    log.info(
        "Starting training: %d epochs, %d steps/epoch, %d markers: %s",
        n_epochs, steps_per_epoch, n_markers, ", ".join(marker_names),
    )
    log.info(
        "Cycle config: %s",
        {c: [m for m in ms if m in marker_names] for c, ms in cycle_config.items()},
    )
    model.train()

    history: Dict[str, List[float]] = {
        "loss": [], "recon": [], "contrast": [], "adv": [], "align": [],
        "cross_batch": [], "lr": [], "grl_lambda": [],
    }

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_contrast = 0.0
        epoch_adv = 0.0
        epoch_align = 0.0
        epoch_cross_batch = 0.0

        # GRL lambda ramp
        if epoch < grl_ramp_epochs:
            grl_lam = epoch / grl_ramp_epochs
        else:
            grl_lam = 1.0

        for step in range(steps_per_epoch):
            idx = sampler.sample()
            if len(idx) < 10:
                continue

            # Build local subgraph (fast: only touches batch nodes)
            edge_local, n_local = adj_index.subgraph(idx)
            edge_local = edge_local.to(device)

            X_batch = X_all[idx].to(device)
            batch_ids = batch_all[idx].to(device)
            sample_ids = sample_all[idx].to(device)

            # Forward
            out = model(X_batch, edge_local, batch_ids, sample_ids, marker_cycles_t, grl_lam)

            # Losses
            loss_recon = huber(out["X_recon"], out["X_corrected"])
            loss_contrast = nt_xent_loss(out["z_proj"], edge_local, tau=tau)
            loss_adv = ce(out["batch_logits"], batch_ids)
            loss_align = quantile_alignment_loss(out["X_recon"], sample_ids)
            loss_cross = cross_batch_nt_xent_loss(
                out["z_proj"], out["X_corrected"], batch_ids,
                tau=tau, n_anchors=cross_batch_anchors, k_positives=cross_batch_k,
            )

            loss = (loss_recon
                    + w_contrast * loss_contrast
                    + w_adv * loss_adv
                    + w_align * loss_align
                    + w_cross_batch * loss_cross)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_contrast += loss_contrast.item()
            epoch_adv += loss_adv.item()
            epoch_align += loss_align.item()
            epoch_cross_batch += loss_cross.item()

            # Per-step progress for long epochs
            if steps_per_epoch > 20 and (step + 1) % max(1, steps_per_epoch // 5) == 0:
                log.info(
                    "  Epoch %3d  step %d/%d  loss=%.4f  recon=%.4f  contrast=%.4f  adv=%.4f  align=%.4f  xbatch=%.4f",
                    epoch + 1, step + 1, steps_per_epoch,
                    loss.item(), loss_recon.item(),
                    loss_contrast.item(), loss_adv.item(), loss_align.item(),
                    loss_cross.item(),
                )

        scheduler.step()
        n_steps = max(steps_per_epoch, 1)

        # Record history
        history["loss"].append(epoch_loss / n_steps)
        history["recon"].append(epoch_recon / n_steps)
        history["contrast"].append(epoch_contrast / n_steps)
        history["adv"].append(epoch_adv / n_steps)
        history["align"].append(epoch_align / n_steps)
        history["cross_batch"].append(epoch_cross_batch / n_steps)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grl_lambda"].append(grl_lam)

        # Log every epoch (not just every 10th)
        log.info(
            "Epoch %3d/%d  loss=%.4f  recon=%.4f  contrast=%.4f  adv=%.4f  align=%.4f  "
            "xbatch=%.4f  grl=%.2f  lr=%.2e  [%d cells/step]",
            epoch + 1, n_epochs,
            history["loss"][-1], history["recon"][-1],
            history["contrast"][-1], history["adv"][-1],
            history["align"][-1], history["cross_batch"][-1],
            grl_lam, optimizer.param_groups[0]["lr"],
            cells_per_step,
        )

    log.info("Training complete. Final loss=%.4f", history["loss"][-1])
    return model, scaler, marker_cycles, history


# ──────────────────────────────────────────────────────────────────────────────
# Post-hoc mode alignment
# ──────────────────────────────────────────────────────────────────────────────


def _find_peaks(
    counts: np.ndarray,
    bins: np.ndarray,
    min_prominence_frac: float = 0.02,
) -> List[float]:
    """Find prominent peaks in a histogram, ordered left to right.

    Returns a list of peak positions (bin centers).  For CyCIF markers:
    - peaks[0] = negative/background population (always leftmost)
    - peaks[1] = positive population (if bimodal)

    Uses Gaussian smoothing + scipy peak detection with prominence filtering.
    """
    counts_smooth = gaussian_filter1d(counts.astype(np.float64), sigma=2)
    max_height = counts_smooth.max()
    if max_height < 1:
        return [(bins[0] + bins[-1]) / 2]

    peaks, _ = find_peaks(
        counts_smooth, prominence=max_height * min_prominence_frac
    )
    if len(peaks) == 0:
        idx = int(counts_smooth.argmax())
        return [(bins[idx] + bins[idx + 1]) / 2]

    return [(bins[p] + bins[p + 1]) / 2 for p in peaks]


def _safe_piecewise_transform(
    vals: np.ndarray,
    src_peaks: List[float],
    dst_peaks: List[float],
) -> np.ndarray:
    """Piecewise linear transform with clamped extrapolation (slope=1).

    - Below the negative peak: pure shift by (dst[0] - src[0])
    - Between negative and positive peaks: linearly scaled to map
      src gap → dst gap (this corrects dynamic range differences)
    - Above the positive peak: pure shift by (dst[1] - src[1]),
      NOT scaled — avoids tail stretching

    Only called when both src and dst have exactly 2 peaks.
    """
    s0, s1 = src_peaks[0], src_peaks[1]
    d0, d1 = dst_peaks[0], dst_peaks[1]

    out = vals.copy()

    # Below negative peak: pure shift
    mask_below = vals <= s0
    out[mask_below] = vals[mask_below] + (d0 - s0)

    # Between peaks: scale to match dynamic range
    src_span = s1 - s0
    dst_span = d1 - d0
    if src_span > 1e-8:
        scale = dst_span / src_span
        mask_mid = (vals > s0) & (vals <= s1)
        out[mask_mid] = d0 + (vals[mask_mid] - s0) * scale
    else:
        # Degenerate: peaks overlap, just shift
        mask_mid = (vals > s0) & (vals <= s1)
        out[mask_mid] = vals[mask_mid] + (d0 - s0)

    # Above positive peak: pure shift (slope=1, NOT scaled)
    mask_above = vals > s1
    out[mask_above] = vals[mask_above] + (d1 - s1)

    return out


def sample_mode_align(
    X: np.ndarray,
    sample_ids: np.ndarray,
    n_bins: int = 200,
    marker_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Per-marker, per-sample peak alignment with safe piecewise correction.

    For each marker, detects peaks in global and per-sample distributions:

    - **Unimodal** (1 peak in global or sample): Pure shift aligning the
      leftmost (negative) peak.  No scaling, no shape distortion.
    - **Bimodal** (2+ peaks in BOTH global and sample): Piecewise linear
      transform that aligns both negative and positive peaks.  Between
      the peaks, values are linearly scaled to match the global dynamic
      range.  Beyond the positive peak, a pure shift is applied (slope=1)
      — this avoids the tail-stretching artifact that occurs when the
      inter-peak scale factor is extrapolated into the tail.

    Args:
        X: (N, M) expression matrix (in log space).
        sample_ids: (N,) per-cell sample labels.
        n_bins: Number of histogram bins for peak estimation.
        marker_names: Optional list of marker names for logging.

    Returns:
        X_aligned: (N, M) peak-aligned expression matrix.
    """
    unique_samples = np.unique(sample_ids)
    n_markers = X.shape[1]

    X_aligned = X.copy()

    for m in range(n_markers):
        col = X[:, m]
        # Use global percentile range for consistent binning across samples
        lo, hi = np.percentile(col, [1, 99])
        if hi - lo < 1e-6:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)

        # Global peaks
        counts_global, _ = np.histogram(col, bins=bins)
        global_peaks = _find_peaks(counts_global, bins)

        n_piecewise = 0
        n_shift = 0

        for s in unique_samples:
            mask = sample_ids == s
            vals = col[mask]
            if len(vals) < 50:
                continue
            counts_s, _ = np.histogram(vals, bins=bins)
            sample_peaks = _find_peaks(counts_s, bins)

            if len(global_peaks) >= 2 and len(sample_peaks) >= 2:
                # Both bimodal: piecewise alignment of first two peaks
                X_aligned[mask, m] = _safe_piecewise_transform(
                    vals, sample_peaks[:2], global_peaks[:2],
                )
                n_piecewise += 1
            else:
                # Unimodal fallback: pure shift on leftmost peak
                shift = global_peaks[0] - sample_peaks[0]
                X_aligned[mask, m] = vals + shift
                n_shift += 1

        n_total = n_piecewise + n_shift
        mname = marker_names[m] if marker_names and m < len(marker_names) else str(m)
        log.info(
            "  %-10s: global_peaks=%d, piecewise=%d/%d, shift=%d/%d",
            mname, len(global_peaks), n_piecewise, n_total, n_shift, n_total,
        )

    return X_aligned


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def normalize_adata(
    adata: ad.AnnData,
    model: SpaNCy,
    scaler: RobustScaler,
    marker_cycles: np.ndarray,
    k_neighbors: int = 15,
    device_str: str = "cpu",
    inference_batch_size: int = 50000,
    mode: str = "affine",
    align_samples: bool = False,
    sample_col: str = "sample_id",
) -> ad.AnnData:
    """Run inference and store results in adata.layers['normalized'].

    Args:
        mode: Correction strategy.
            "affine"   — (default) Apply only the learned CycleDegradationModel
                         correction (per-batch, per-marker gamma/beta).  This is
                         a pure shift+scale that **perfectly preserves distribution
                         shape** (bimodality, positive populations, tails).
                         The GNN/decoder are used only during training as a
                         scaffold to learn good gamma/beta values.
            "residual" — Full pipeline: affine correction + GNN encoder → residual
                         decoder.  Better batch alignment (kBET) but may distort
                         distribution shape.
        align_samples: If True, apply post-hoc per-marker, per-sample
            mode (histogram peak) alignment after the primary correction.
            For each marker, each sample's dominant peak is shifted to match
            the global peak.  Unlike median alignment, this is robust to
            bimodal markers (e.g. ECAD) where the median depends on the
            positive cell fraction (biology, not batch effect).
            Default False — the model's per-sample affine correction usually
            handles this; enable for additional refinement.
        sample_col: obs column identifying individual samples for alignment.
            Default "sample_id".
    """
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    X_raw = np.asarray(adata.X)
    if sp.issparse(adata.X):
        X_raw = adata.X.toarray()

    X_log = np.log1p(np.clip(X_raw, 0, None))
    X_scaled = scaler.transform(X_log).astype(np.float32)

    batch_cats = adata.obs["batch"].astype("category")
    batch_codes = batch_cats.cat.codes.values.astype(np.int64)

    # Encode sample labels (must match training encoding)
    s_col = None
    for col in (sample_col, "sample_id", "sample", "Sample", "patient_id", "patient"):
        if col in adata.obs.columns:
            s_col = col
            break
    if s_col is None:
        s_col = "batch"
    sample_cats = adata.obs[s_col].astype("category")
    sample_codes = sample_cats.cat.codes.values.astype(np.int64)

    marker_cycles_t = torch.tensor(marker_cycles, dtype=torch.long, device=device)
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    batch_all = torch.tensor(batch_codes, dtype=torch.long)
    sample_all = torch.tensor(sample_codes, dtype=torch.long)

    n_cells = adata.n_obs
    n_markers = adata.n_vars
    X_norm_scaled = np.zeros((n_cells, n_markers), dtype=np.float32)

    if mode == "affine":
        # ── Affine-only: just apply learned gamma/beta ──────────────────
        # No spatial graph needed — much faster, perfectly shape-preserving.
        log.info("Normalizing with affine-only correction (shape-preserving)...")

        for chunk_i in range(max(1, math.ceil(n_cells / inference_batch_size))):
            start = chunk_i * inference_batch_size
            end = min(start + inference_batch_size, n_cells)

            X_chunk = X_all[start:end].to(device)
            batch_chunk = batch_all[start:end].to(device)
            sample_chunk = sample_all[start:end].to(device)

            X_out = model.cycle_model.correct(X_chunk, batch_chunk, sample_chunk, marker_cycles_t)
            X_norm_scaled[start:end] = X_out.cpu().numpy()

            if (chunk_i + 1) % 10 == 0 or chunk_i == 0:
                log.info("Inference chunk %d/%d", chunk_i + 1,
                         max(1, math.ceil(n_cells / inference_batch_size)))

    elif mode == "residual":
        # ── Full pipeline: affine + GNN + residual decoder ──────────────
        coords = get_spatial_coords(adata)
        scene_ids = get_scene_ids(adata)

        log.info("Normalizing with full residual pipeline...")
        log.info("Building inference spatial graph...")
        edge_index_np = build_knn_graph(coords, scene_ids, k=k_neighbors)

        all_indices = np.arange(n_cells)
        n_chunks = max(1, math.ceil(n_cells / inference_batch_size))

        for chunk_i in range(n_chunks):
            start = chunk_i * inference_batch_size
            end = min(start + inference_batch_size, n_cells)
            idx = all_indices[start:end]

            edge_local, _ = build_subgraph(edge_index_np, idx)
            edge_local = edge_local.to(device)

            X_chunk = X_all[idx].to(device)
            batch_chunk = batch_all[idx].to(device)
            sample_chunk = sample_all[idx].to(device)

            X_out = model.normalize(X_chunk, edge_local, batch_chunk, sample_chunk, marker_cycles_t)
            X_norm_scaled[start:end] = X_out.cpu().numpy()

            if (chunk_i + 1) % 10 == 0 or chunk_i == 0:
                log.info("Inference chunk %d/%d", chunk_i + 1, n_chunks)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'affine' or 'residual'.")

    # Inverse transform: unscale → log space
    X_norm_log = scaler.inverse_transform(X_norm_scaled)

    # Per-sample MODE alignment IN LOG SPACE (before expm1).
    # Uses histogram peak (mode) instead of median — robust to bimodal
    # markers where the median depends on positive cell fraction (biology).
    # Shifting in log space = multiplicative scaling in original space,
    # which cannot push values to zero or create clipping artifacts.
    if align_samples:
        s_col = None
        for col in (sample_col, "sample", "Sample", "patient_id", "patient"):
            if col in adata.obs.columns:
                s_col = col
                break
        if s_col is not None:
            sample_ids = adata.obs[s_col].values
            n_unique = len(np.unique(sample_ids))
            if n_unique > 1:
                log.info(
                    "Applying per-sample mode alignment in log space "
                    "across %d samples (col='%s')...",
                    n_unique, s_col,
                )
                mnames = list(adata.var_names) if adata is not None else None
                X_norm_log = sample_mode_align(
                    X_norm_log, sample_ids, marker_names=mnames,
                )
            else:
                log.info("Only 1 sample found — skipping sample alignment.")
        else:
            log.warning(
                "align_samples=True but no sample column found "
                "(tried: %s, sample, Sample, patient_id). Skipping.",
                sample_col,
            )

    # expm1 inverts log1p; clip at 0 (physical constraint)
    X_norm = np.expm1(X_norm_log)
    X_norm = np.clip(X_norm, 0, None)

    adata.layers["normalized"] = X_norm
    log.info(
        "Normalization complete [mode=%s, quantile_align=%s]. "
        "Stored in adata.layers['normalized'] "
        "(min=%.4f, max=%.4f, mean=%.4f)",
        mode, align_samples,
        X_norm.min(), X_norm.max(), X_norm.mean(),
    )
    return adata


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="spancy",
        description="SpaNCy: Spatial Neighborhood Contrastive CyCIF Normalizer",
    )
    p.add_argument("--input", "-i", required=True, help="Input .h5ad file path")
    p.add_argument("--output", "-o", required=True, help="Output .h5ad file path")
    p.add_argument("--epochs", "-e", type=int, default=100, help="Training epochs (default: 100)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--k_neighbors", "-k", type=int, default=15, help="k-NN neighbors (default: 15)")
    p.add_argument(
        "--cycle_config",
        type=str,
        default=None,
        help="JSON string or file path for cycle->marker mapping. "
        'Format: {"0": ["DAPI"], "1": ["EPCAM","CD56",...], ...}',
    )
    p.add_argument("--batch_size", type=int, default=16000, help="Cells per training step (default: 16000)")
    p.add_argument("--cluster_size", type=int, default=500, help="Spatial cluster size (default: 500)")
    p.add_argument("--device", "-d", type=str, default="cpu", help="Device: cpu or cuda (default: cpu)")
    p.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs (default: 5)")
    p.add_argument("--grl_ramp_epochs", type=int, default=30, help="GRL ramp epochs (default: 30)")
    p.add_argument("--w_contrast", type=float, default=0.5, help="Contrastive loss weight (default: 0.5)")
    p.add_argument("--w_adv", type=float, default=0.3, help="Adversarial loss weight (default: 0.3)")
    p.add_argument("--w_align", type=float, default=0.5, help="Quantile alignment weight (default: 0.5)")
    p.add_argument("--w_cross_batch", type=float, default=0.5, help="Cross-batch contrastive weight (default: 0.5)")
    p.add_argument("--tau", type=float, default=0.1, help="NT-Xent temperature (default: 0.1)")
    p.add_argument("--cross_batch_anchors", type=int, default=512, help="Anchor cells per step for cross-batch contrastive (default: 512)")
    p.add_argument("--cross_batch_k", type=int, default=5, help="Cross-batch positives per anchor (default: 5)")
    return p.parse_args(argv)


def load_cycle_config(config_arg: Optional[str]) -> Dict[int, List[str]]:
    """Parse cycle config from JSON string, file path, or return default."""
    if config_arg is None:
        return DEFAULT_CYCLE_CONFIG

    # Try as file path first
    try:
        with open(config_arg, "r") as f:
            raw = json.load(f)
    except (FileNotFoundError, OSError):
        raw = json.loads(config_arg)

    return {int(k): v for k, v in raw.items()}


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cycle_config = load_cycle_config(args.cycle_config)

    adata = load_adata(args.input)

    model, scaler, marker_cycles, history = train(
        adata,
        cycle_config=cycle_config,
        n_epochs=args.epochs,
        lr=args.lr,
        k_neighbors=args.k_neighbors,
        cluster_size=args.cluster_size,
        cells_per_step=args.batch_size,
        device_str=args.device,
        warmup_epochs=args.warmup_epochs,
        grl_ramp_epochs=args.grl_ramp_epochs,
        w_contrast=args.w_contrast,
        w_adv=args.w_adv,
        w_align=args.w_align,
        w_cross_batch=args.w_cross_batch,
        tau=args.tau,
        cross_batch_anchors=args.cross_batch_anchors,
        cross_batch_k=args.cross_batch_k,
    )

    adata = normalize_adata(
        adata,
        model,
        scaler,
        marker_cycles,
        k_neighbors=args.k_neighbors,
        device_str=args.device,
    )

    log.info("Saving to %s", args.output)
    adata.write_h5ad(args.output)
    log.info("Done. Output: %s (%d cells x %d markers)", args.output, adata.n_obs, adata.n_vars)


if __name__ == "__main__":
    main()
