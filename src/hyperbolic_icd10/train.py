"""Training loop used for every number in the paper draft (Tables 1–5).

Port of ``train_logged`` from ``notebooks/09_paper_experiments.ipynb`` (which
is itself ``train3`` from notebook 08 with the checkpoint-selection removed).
Requires ``torch`` and ``geoopt``; everything else in the package is numpy.

The loss, with temperature ``scale`` (the paper's τ; the notebooks' ``x``):

    d_ij  = dist(e_i, e_j)                      Poincaré ball, c = 1
    d'_ij = d_ij · exp(α · ½(κ_i + κ_j))        only in ``graded`` mode
    L     = mean_i [ τ·d'_{i,pos} + logsumexp(−τ·d'_{i,·}) ]  over {pos} ∪ negatives
          + β · mean relu(‖e_i‖ − ‖e_pos‖ + 0.05)             hierarchy term, β = 1

Optimiser: RiemannianSGD, lr 50 with a 10-epoch burn-in at lr/100, batch 1024,
K = 50 negatives resampled per batch, ROOT pinned at the origin.  Euclidean
mode uses squared distance, Adam lr 0.1, no hierarchy term.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .data import Tree
from .evalsets import EvalSets
from .metrics import eval_all


def graded_kappa(tree: Tree, scale: float = 1.0) -> np.ndarray:
    """The 'graded' per-node curvature feature: (log(1+b))² min-max scaled to [−1, 1]."""
    k = np.log1p(tree.branching_factor.astype(float)) ** 2
    k = 2 * (k - k.min()) / (k.max() - k.min() + 1e-9) - 1
    return (k * scale).astype(np.float32)


def sample_negs(tree: Tree, anchors: np.ndarray, K: int) -> np.ndarray:
    """K uniform negatives per anchor, rejecting the anchor and its graph neighbours."""
    N = tree.N
    out = np.random.randint(0, N, size=(len(anchors), K))
    for i, a in enumerate(anchors):
        for j in range(K):
            while out[i, j] == a or (a, out[i, j]) in tree.connected:
                out[i, j] = np.random.randint(0, N)
    return out


def train_logged(tree: Tree, sets: EvalSets, mode: str = "const", scale: float = 1.0,
                 dim: int = 10, alpha: float = -2.0, epochs: int = 1500, lr: float = 50,
                 K: int = 50, seed: int = 0, beta: float = 1.0, log_every: int = 300,
                 tag: Optional[str] = None, kappa_override: Optional[np.ndarray] = None,
                 device: Optional[str] = None, verbose: bool = True,
                 save_positions: Optional[Callable[[np.ndarray], None]] = None) -> dict:
    """Train once and return the full metric trajectory (no checkpoint selection).

    ``mode``: ``const`` (Poincaré, constant curvature), ``graded`` (per-node
    multiplicative curvature feature — a *scale* change, see README), or
    ``euclid``.
    """
    import torch
    import torch.nn as nn
    import geoopt

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    tag = tag or f"{mode}_x{scale}_d{dim}_s{seed}"
    N, ROOT = tree.N, tree.ROOT
    kf = torch.tensor(kappa_override if kappa_override is not None else graded_kappa(tree),
                      dtype=torch.float32, device=device)

    if mode == "euclid":
        emb = nn.Parameter(torch.empty(N, dim, device=device).uniform_(-1e-3, 1e-3))
        opt = torch.optim.Adam([emb], lr=0.1)
        params = emb
    else:
        manifold = geoopt.PoincareBall(c=1.0)
        params = geoopt.ManifoldParameter(
            torch.empty(N, dim).uniform_(-1e-3, 1e-3), manifold=manifold).to(device)
        opt = geoopt.optim.RiemannianSGD([params], lr=lr)
        with torch.no_grad():
            params.data[ROOT] = 0.0

    E = np.array(tree.edges_idx)
    traj, t0 = [], time.time()
    for ep in range(epochs):
        if mode != "euclid":
            eff = lr * 0.01 if ep < 10 else lr
            for pg in opt.param_groups:
                pg["lr"] = eff
        perm = np.random.permutation(len(E))
        for bs in range(0, len(E), 1024):
            b = E[perm[bs:bs + 1024]]
            a = torch.tensor(b[:, 0]).to(device)
            p = torch.tensor(b[:, 1]).to(device)
            ng = torch.tensor(sample_negs(tree, b[:, 0], K)).to(device)
            ea, ep_, en = params[a], params[p], params[ng]
            if mode == "euclid":
                dp = ((ea - ep_) ** 2).sum(-1) * scale
                dn = ((ea.unsqueeze(1) - en) ** 2).sum(-1) * scale
                loss = (dp + torch.logsumexp(-torch.cat([dp.unsqueeze(1), dn], 1), 1)).mean()
            else:
                bp = manifold.dist(ea, ep_)
                bn = manifold.dist(ea.unsqueeze(1), en)
                if mode == "graded":
                    bp = bp * torch.exp(alpha * 0.5 * (kf[a] + kf[p]))
                    bn = bn * torch.exp(alpha * 0.5 * (kf[a].unsqueeze(1) + kf[ng]))
                dp, dn = bp * scale, bn * scale
                nk = (dp + torch.logsumexp(-torch.cat([dp.unsqueeze(1), dn], 1), 1)).mean()
                an, pn = ea.norm(dim=-1), ep_.norm(dim=-1)
                loss = nk + beta * torch.relu(an - pn + 0.05).mean()
            opt.zero_grad()
            loss.backward()
            if mode != "euclid" and params.grad is not None:
                params.grad[ROOT] = 0.0
            opt.step()

        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            pos = params.detach().cpu().numpy()
            if np.isnan(pos).any():
                break
            m = eval_all(pos, sets, tree.nbrs, euclidean=(mode == "euclid"))
            if m:
                m["epoch"] = ep + 1
                traj.append(m)

    if save_positions is not None:
        save_positions(params.detach().cpu().numpy())
    out = {"tag": tag, "mode": mode, "scale": scale, "dim": dim, "seed": seed,
           "alpha": alpha, "beta": beta, "epochs": epochs, "lr": lr, "K": K,
           "trajectory": traj, "minutes": (time.time() - t0) / 60}
    if verbose and traj:
        f = traj[-1]
        print(f"  {tag}: final MAP_test {f['MAP_test']:.4f}  closMRR {f['closMRR']:.4f}  "
              f"dist {f['distortion']:.4f}  ({out['minutes']:.1f}m)")
    return out


def select(traj, rule: str = "final"):
    """``final`` (paper) | ``val`` (best MAP_val) | ``test`` (best MAP_test — the old flawed rule)."""
    if not traj:
        return None
    if rule == "final":
        return traj[-1]
    key = "MAP_val" if rule == "val" else "MAP_test"
    return max(traj, key=lambda m: m[key])
