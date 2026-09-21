# ZiPert

ZiPert is a Python package for fitting negative-binomial count models for
differential expression analysis, with support for zero inflation and
gene-set-level shared dropout.

The package name is currently `deg_zinb`, so imports use `deg_zinb` even when
the repository is named `ZiPert`.

## Models

ZiPert implements PyTorch-based GLMs for single-cell count data:

- `NBGLM`: gene-wise negative binomial GLM
- `ZINBGLM`: gene-wise zero-inflated negative binomial GLM
- `MZINBGLM`: marginalized zero-inflated negative binomial GLM
- `CZINBGLM`: consensus/shared-dropout ZINB for a set of genes

The package also includes Wald and likelihood-ratio testing helpers for fitted
models.

## Installation

From a local checkout:

```bash
git clone git@github.com:thewonlab/ZiPert.git
cd ZiPert
pip install -e .
```

Main dependencies are declared in `pyproject.toml`:

- `anndata`
- `numpy`
- `pandas`
- `torch`

## Quick Start

### Prepare Cell Ranger Outputs

Install the optional reader with `pip install -e '.[cellranger]'`.
`read_cellranger` accepts an `outs` directory, a filtered feature matrix H5,
or a compressed Cell Ranger v3+ MTX directory. It uses Scanpy's
[10x readers](https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html)
with all feature types enabled. Both Gene Expression and CRISPR Guide Capture
features are required; it does not run Cell Ranger or process FASTQ files.

```python
from deg_zinb import read_cellranger, prepare_zipert_inputs, fit_glm

counts = read_cellranger({
    "inlet1": "/path/to/inlet1/outs",
    "inlet2": "/path/to/inlet2/outs",
})
print(counts.uns["guide_ids"])  # Exact feature IDs to use below

adata, X_design = prepare_zipert_inputs(
    counts,
    target_guides=["1FOXP2positive", "2FOXP2positive"],
    control_guides=["1GFPnon-targeting", "2NTnon-targeting"],
    min_guide_umis=5,
    min_genes=200,
    max_percent_mt=20,
)
result = fit_glm(adata, genes=["ENSG00000128573"], X_design=X_design)

adata.write_h5ad("zipert_input.h5ad")
X_design.to_csv("X_design.csv", index_label="cell_id")
```

Guide IDs and gene IDs in this example must be replaced with IDs present in
your feature reference. Gene IDs are `adata.var_names`; symbols are in
`adata.var['gene_symbols']`. Original barcodes and inlet names are in `obs`;
cell IDs are inlet-prefixed to avoid collisions. Raw sparse RNA counts stay
in `X`, and sparse guide counts stay in `obsm['guide_counts']`.

Assignment uses the existing example's rule: exactly one guide must reach the
UMI threshold (default 5). Cells with multiple qualifying guides, no qualifying
guide, or guides outside the requested comparison are excluded. This is a
custom threshold assignment, not Cell Ranger's protospacer calls.

The default design contains an intercept, target=1/control=0, guide detection
count, log1p guide/RNA UMI totals, mitochondrial percentage, and inlet dummies.
Pass `covariates=()` for just group/intercept plus inlet, or choose a subset of
numeric `obs` columns. Rank-deficient designs raise an error rather than fit
unidentifiable effects. QC thresholds default to no filtering except zero-RNA
cells; the example thresholds above are optional. No cell-type annotation or
glutamatergic-neuron selection is performed. Subset `counts` before preparation
when needed. Inlets must share the same feature reference. Default mitochondrial
prefix is `MT-`; set `mito_prefix` for other naming conventions.

When reloading the CSV, use `pd.read_csv("X_design.csv", index_col="cell_id")`
and preserve its alignment with `adata.obs_names`.

### Fit Prepared Inputs

ZiPert expects an `AnnData` object, a list of genes, and a design matrix whose
rows match `adata.obs`.

```python
import anndata as ad
import numpy as np
import pandas as pd
import torch

from deg_zinb import fit_glm
from deg_zinb.torch_backend.fit import FitConfig
from deg_zinb.torch_backend.model import GLMConfig

adata = ad.AnnData(
    X=np.array([
        [1, 0],
        [2, 1],
        [0, 0],
        [4, 2],
        [5, 1],
        [6, 3],
        [3, 0],
        [7, 4],
    ], dtype=float),
    obs=pd.DataFrame(
        {"group": [0, 0, 0, 0, 1, 1, 1, 1]},
        index=[f"cell{i}" for i in range(8)],
    ),
    var=pd.DataFrame(index=["GENE1", "GENE2"]),
)

# Example design matrix. Include an intercept if you want one.
X_design = pd.DataFrame({
    "intercept": 1.0,
    "group": adata.obs["group"].astype(float),
}, index=adata.obs_names)

result = fit_glm(
    adata=adata,
    genes=["GENE1", "GENE2"],
    X_design=X_design,
    model="nb",               # "nb", "zinb", or "mzinb"
    offset_key=None,           # e.g. "log_library_size" if stored in adata.obs
    fit_cfg=FitConfig(method="lbfgs", max_iter=50),
    glm_cfg=GLMConfig(ridge=0.0),
    device="cpu",
    seed=1,
    dtype=torch.float64,
    n_jobs=1,
)
```

The returned object behaves like a dictionary keyed by gene:

```python
result["GENE1"]["model_kind"]
result["GENE1"]["neg_log_likelihood"]
result["GENE1"]["state_dict"]
```

## Wald Tests

After fitting, call `add_wald()` to compute Hessian-based standard errors and
summary statistics.

```python
result.add_wald()
result.wald_df.head()
```

Use `robust=True` for sandwich covariance:

```python
result.add_wald(robust=True)
```

Terms are reported separately for the count and zero-inflation components, for
example:

- `count:group`
- `zero:group`
- `log(theta)`

## Likelihood-Ratio Tests

Use `add_lrt()` to compare a full model against a reduced model with one or
more design terms removed.

```python
result.add_lrt("group")
result.lrt_df
```

For NB models, dropping one term uses 1 degree of freedom. For ZINB and MZINB,
the same term is dropped from both the count and zero components, so the test
uses 2 degrees of freedom per term.

## Consensus ZINB

`CZINBGLM` models a gene set with gene-specific count parameters and a shared
cell-level dropout probability.

```python
from deg_zinb import fit_czinb_glm

czinb_model = fit_czinb_glm(
    adata=adata,
    genes=["GENE1", "GENE2", "GENE3"],
    X_design=X_design,
    fit_cfg=FitConfig(method="lbfgs", max_iter=200),
    glm_cfg=GLMConfig(ridge=0.0),
    device="cpu",
    seed=1,
    dtype=torch.float64,
)
```

You can also fit multiple gene sets. If no gene sets are supplied, genes can be
clustered by sparsity.

```python
from deg_zinb import fit_czinb_gene_sets

gene_sets = {
    "low_sparsity": ["GENE1", "GENE2"],
    "high_sparsity": ["GENE3", "GENE4"],
}

set_result = fit_czinb_gene_sets(
    adata=adata,
    gene_sets=gene_sets,
    X_design=X_design,
    fit_cfg=FitConfig(method="lbfgs", max_iter=200),
    min_genes=1,
)
```

For CZINB likelihood-ratio testing:

```python
from deg_zinb import lrt_czinb

lrt_out = lrt_czinb(
    adata=adata,
    genes=["GENE1", "GENE2"],
    X_count_design=X_design,
    lrt_target="group",
    mode="count_only",  # or "full_refit"
)

lrt_out["table"]
```

## Notes on Inputs

- `adata[:, genes].X` should contain non-negative count data.
- `X_design` should be a pandas `DataFrame` when using named terms in LRT.
- If using an offset, store it in `adata.obs` and pass its column name as
  `offset_key`.
- For reproducible CPU fits, set `seed`; deterministic GPU behavior depends on
  PyTorch and CUDA settings.

## Testing

Run the smoke tests from the repository root:

```bash
pytest tests/test_smoke.py
```

Large local result artifacts are intentionally not tracked in git. See
`.gitignore` for excluded cache and experiment-output paths.
