"""Prepare Cell Ranger feature counts for ZiPert without normalizing counts."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _counts(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if (not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
            or (matrix.data != np.floor(matrix.data)).any()):
        raise ValueError("Expected finite, non-negative integer UMI counts.")
    return matrix


def read_cellranger(paths, *, mito_prefix="MT-", guide_feature_type="CRISPR Guide Capture"):
    """Read filtered Cell Ranger H5/MTX counts, or a mapping of inlet -> path.

    Paths may point to an outs directory, filtered_feature_bc_matrix directory,
    or filtered_feature_bc_matrix.h5. Requires the optional ``cellranger`` extra.
    Gene and guide feature IDs must be unique and identical across inlets.
    Gene symbols are retained in var['gene_symbols']; var_names are gene IDs.
    Guide counts remain sparse in obsm['guide_counts'], with IDs/names in uns.
    No cell-type selection, normalization, or automatic QC filtering is applied.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError('Install Cell Ranger support with pip install "deg_zinb[cellranger]".') from exc

    samples = paths if isinstance(paths, Mapping) else {"inlet1": paths}
    if not samples or any(not isinstance(k, str) or not k for k in samples):
        raise ValueError("Provide at least one inlet with a non-empty string name.")
    parts = []
    reference = None
    for inlet, source in samples.items():
        path = Path(source)
        if path.is_dir() and (path / "filtered_feature_bc_matrix.h5").is_file():
            path = path / "filtered_feature_bc_matrix.h5"
        elif path.is_dir() and (path / "filtered_feature_bc_matrix").is_dir():
            path = path / "filtered_feature_bc_matrix"
        if path.is_dir():
            data = sc.read_10x_mtx(path, var_names="gene_ids", gex_only=False, cache=False)
            ids = data.var_names.astype(str)
            symbols = data.var["gene_symbols"].astype(str).to_numpy()
        else:
            data = sc.read_10x_h5(path, gex_only=False)
            ids = pd.Index(data.var["gene_ids"].astype(str))
            symbols = data.var_names.astype(str).to_numpy()
        if not data.obs_names.is_unique:
            raise ValueError(f"Duplicate cell barcodes in inlet {inlet}.")
        types = data.var["feature_types"].astype(str).to_numpy()
        selected = (types == "Gene Expression") | (types == guide_feature_type)
        data = data[:, selected].copy()
        data.var_names = ids[selected]
        data.var["gene_symbols"] = symbols[selected]
        if not data.var_names.is_unique:
            raise ValueError(f"Duplicate feature IDs in inlet {inlet}.")
        if reference is None:
            reference = data.var.copy()
        else:
            if set(data.var_names) != set(reference.index):
                raise ValueError("Inlets must use the same gene and guide feature reference.")
            data = data[:, reference.index].copy()
            for column in ("feature_types", "gene_symbols"):
                if not np.array_equal(data.var[column].astype(str), reference[column].astype(str)):
                    raise ValueError(f"Inconsistent {column} across inlets.")
        rna_mask = data.var["feature_types"].eq("Gene Expression").to_numpy()
        guide_mask = data.var["feature_types"].eq(guide_feature_type).to_numpy()
        if not rna_mask.any() or not guide_mask.any():
            raise ValueError("Both Gene Expression and guide capture features are required.")
        rna = data[:, rna_mask].copy()
        rna.X = _counts(rna.X)
        guides = _counts(data[:, guide_mask].X)
        rna.obsm["guide_counts"] = guides
        rna.uns["guide_ids"] = data.var_names[guide_mask].to_numpy(dtype=str)
        rna.uns["guide_names"] = data.var.loc[guide_mask, "gene_symbols"].to_numpy(dtype=str)
        rna.obs["barcode"] = rna.obs_names.astype(str)
        rna.obs["inlet"] = inlet
        rna.obs_names = pd.Index([f"{inlet}:{barcode}" for barcode in rna.obs_names])
        total = np.asarray(rna.X.sum(axis=1)).ravel()
        mito = rna.var["gene_symbols"].str.startswith(mito_prefix).to_numpy()
        mito_total = np.asarray(rna.X[:, mito].sum(axis=1)).ravel()
        rna.obs["response_n_umis"] = total
        rna.obs["response_n_nonzero"] = rna.X.getnnz(axis=1)
        rna.obs["percent_mt"] = np.divide(
            100 * mito_total, total, out=np.zeros_like(total), where=total > 0
        )
        rna.obs["grna_n_umis"] = np.asarray(guides.sum(axis=1)).ravel()
        rna.obs["grna_n_nonzero"] = guides.getnnz(axis=1)
        parts.append(rna)
    result = ad.concat(parts, merge="same", uns_merge="same")
    if not result.obs_names.is_unique:
        raise ValueError("Inlet names and barcodes produce duplicate cell IDs.")
    return result


def prepare_zipert_inputs(
    adata, *, target_guides, control_guides, min_guide_umis=5,
    min_genes=0, max_percent_mt=100,
    covariates=("grna_n_nonzero", "grna_n_umis", "response_n_umis", "percent_mt"),
    include_inlet=True,
):
    """Return (count AnnData, design DataFrame) for one target/control comparison.

    Guide lists contain exact feature IDs from uns['guide_ids'], not substrings.
    Exactly one guide must have >= min_guide_umis; multiplets and unassigned
    cells are excluded. All guides participate in multiplet detection.
    QC includes >= min_genes and <= max_percent_mt, and excludes zero-RNA cells.
    The design includes an intercept, target=1/control=0, selected covariates
    (log1p for UMI totals), and optional inlet dummies with one reference inlet.
    QC metrics in obs remain on their original scale. Rank-deficient designs
    are rejected; choose covariates suitable for the experiment.
    """
    if not np.isfinite(min_guide_umis) or min_guide_umis < 1:
        raise ValueError("min_guide_umis must be finite and >= 1.")
    if not np.isfinite(min_genes) or min_genes < 0:
        raise ValueError("min_genes must be finite and >= 0.")
    if not np.isfinite(max_percent_mt) or not 0 <= max_percent_mt <= 100:
        raise ValueError("max_percent_mt must be between 0 and 100.")
    if isinstance(target_guides, str) or isinstance(control_guides, str):
        raise TypeError("target_guides and control_guides must be lists of feature IDs.")
    target, control = set(target_guides), set(control_guides)
    if not target or not control or target & control:
        raise ValueError("Target/control guide sets must be non-empty and disjoint.")
    ids = np.asarray(adata.uns["guide_ids"], dtype=str)
    if len(set(ids)) != len(ids):
        raise ValueError("Guide IDs must be unique.")
    missing = (target | control) - set(ids)
    if missing:
        raise ValueError(f"Unknown guide IDs: {sorted(missing)}")
    guides = _counts(adata.obsm["guide_counts"])
    if guides.shape != (adata.n_obs, len(ids)) or not adata.obs_names.is_unique:
        raise ValueError("Guide matrix dimensions or cell IDs are invalid.")
    detected = guides >= min_guide_umis
    n_detected = np.asarray(detected.sum(axis=1)).ravel()
    assigned = ids[np.asarray(guides.argmax(axis=1)).ravel()]
    obs = adata.obs
    keep = ((n_detected == 1) & np.isin(assigned, list(target | control))
            & (obs["response_n_umis"].to_numpy() > 0)
            & (obs["response_n_nonzero"].to_numpy() >= min_genes)
            & (obs["percent_mt"].to_numpy() <= max_percent_mt))
    out = adata[keep].copy()
    out.X = _counts(out.X)
    out.obs["guide_id"] = assigned[keep]
    out.obs["group"] = np.isin(assigned[keep], list(target)).astype(float)
    if out.obs["group"].nunique() != 2:
        raise ValueError("Both target and control cells must remain after filtering.")
    design = pd.DataFrame({"Intercept": 1.0, "group": out.obs["group"]}, index=out.obs_names)
    if isinstance(covariates, str) or len(set(covariates)) != len(covariates):
        raise ValueError("covariates must be a sequence of distinct column names.")
    for column in covariates:
        if column in design or column == "inlet":
            raise ValueError(f"Reserved design column: {column}")
        values = pd.to_numeric(out.obs[column], errors="raise").astype(float)
        if column in ("grna_n_umis", "response_n_umis"):
            values = np.log1p(values)
        design[column] = values
    if include_inlet:
        inlet = pd.Categorical(out.obs["inlet"], categories=sorted(out.obs["inlet"].unique()))
        dummies = pd.get_dummies(inlet, prefix="inlet", drop_first=True, dtype=float)
        dummies.index = out.obs_names
        design = design.join(dummies)
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError("Design covariates contain missing or non-finite values.")
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError("Design is rank deficient: remove constant/redundant covariates "
                         "or check whether group is confounded with inlet.")
    return out, design
