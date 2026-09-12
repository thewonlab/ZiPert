from __future__ import annotations
import numpy as np
from typing import Optional
import torch
def anndata_to_tensors(
    adata,
    gene: str,
    X_design: np.ndarray,
    offset_key: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    adata: AnnData
    gene: var_names entry
    X_design: (n_cells, p) design matrix (numpy-like)
    offset_key: adata.obs column with log-offset (e.g., log library size)
    device: "cpu" | "cuda" | None
    """

    # --- y (counts) ---
    y = adata[:, [gene]].X
    if hasattr(y, "toarray"):   # sparse
        y = y.toarray()
    y = np.asarray(y).reshape(-1).astype(np.float64, copy=False)

    if np.isnan(y).any():
        raise ValueError(f"y has NaNs for gene={gene}")
    if (y < 0).any():
        raise ValueError(f"y has negative values for gene={gene}")

    # --- X design ---
    X = np.asarray(X_design, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X_design shape {X.shape} incompatible with y length {y.shape[0]}")

    # --- offset ---
    off_t = None
    if offset_key is not None:
        log_offset = np.asarray(adata.obs[offset_key], dtype=np.float64).reshape(-1)
        if log_offset.shape[0] != y.shape[0]:
            raise ValueError(f"offset length {log_offset.shape[0]} != y length {y.shape[0]}")
        if np.isnan(log_offset).any():
            raise ValueError(f"offset has NaNs (offset_key={offset_key})")
        off_t = torch.from_numpy(log_offset)

    y_t = torch.from_numpy(y)
    X_t = torch.from_numpy(X)

    if device is not None:
        y_t = y_t.to(device)
        X_t = X_t.to(device)
        if off_t is not None:
            off_t = off_t.to(device)

    return y_t, X_t, off_t

def anndata_to_matrix_tensors(adata, genes, X_design, offset_key=None):
    # Y: (n, G)
    Y = adata[:, genes].X
    if not isinstance(Y, np.ndarray):
        Y = Y.toarray()

    Y = torch.tensor(Y, dtype=torch.float64)

    # X: (n, p)
    X = torch.tensor(X_design.values, dtype=torch.float64)

    log_offset = None
    if offset_key is not None:
        offset = adata.obs[offset_key].values
        log_offset = torch.tensor(offset, dtype=torch.float64)

    return Y, X, log_offset