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

ZiPert expects an `AnnData` object, a list of genes, and a design matrix whose
rows match `adata.obs`.

```python
import pandas as pd
import torch

from deg_zinb import fit_glm
from deg_zinb.torch_backend.fit import FitConfig
from deg_zinb.torch_backend.model import GLMConfig

# Example design matrix. Include an intercept if you want one.
X_design = pd.DataFrame({
    "intercept": 1.0,
    "group": adata.obs["group"].astype(float),
})

result = fit_glm(
    adata=adata,
    genes=["GENE1", "GENE2"],
    X_design=X_design,
    model="nb",               # "nb", "zinb", or "mzinb"
    offset_key=None,           # e.g. "log_library_size" if stored in adata.obs
    fit_cfg=FitConfig(method="lbfgs", max_iter=200),
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
