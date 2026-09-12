from __future__ import annotations

import warnings
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from deg_zinb.api import FitConfig, GLMConfig, fit_glm

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
OUT_DIR = Path("tests/Results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

adata = anndata.read_h5ad("tests/Results/adata_sub.h5ad")
X_design = pd.read_csv("tests/Results/X_design.csv", index_col=0)

Y = adata.X
if sp.issparse(Y):
    Y = Y.toarray()
else:
    Y = np.asarray(Y)

zero_frac = (Y == 0).mean(axis=0)
mean_expr = Y.mean(axis=0)
detected_n = (Y > 0).sum(axis=0)
gene_names = np.asarray(adata.var_names)

gene_df = pd.DataFrame(
    {
        "gene": gene_names,
        "zero_frac": zero_frac,
        "mean_expr": mean_expr,
        "detected_n": detected_n,
    }
)
gene_df["sparsity_bin"] = pd.cut(
    gene_df["zero_frac"],
    bins=[-0.001, 0.70, 0.95, 1.001],
    labels=["low", "mid", "high"],
)

# Selection does not use group labels or logFC.
# Keep genes with enough observed counts to make LRT meaningful.
candidate_df = gene_df[gene_df["detected_n"] >= 25].copy()
selected_frames = []
for sparsity_bin in ["low", "mid", "high"]:
    sub = candidate_df[candidate_df["sparsity_bin"] == sparsity_bin].copy()
    if len(sub) == 0:
        continue

    expr_rank = sub["mean_expr"].rank(method="first")
    sub["expr_bin"] = pd.qcut(expr_rank, q=3, labels=["low_expr", "mid_expr", "high_expr"])

    for expr_bin in ["low_expr", "mid_expr", "high_expr"]:
        cell = sub[sub["expr_bin"] == expr_bin].copy()
        if len(cell) == 0:
            continue
        take_n = min(2, len(cell))
        take_idx = RNG.choice(cell.index.to_numpy(), size=take_n, replace=False)
        selected_frames.append(cell.loc[take_idx])

selected_df = pd.concat(selected_frames, axis=0).drop_duplicates(subset=["gene"]).copy()
selected_df = selected_df.sort_values(["sparsity_bin", "mean_expr", "gene"]).reset_index(drop=True)

if "FOXP2" not in set(selected_df["gene"]):
    foxp2_row = gene_df[gene_df["gene"] == "FOXP2"].copy()
    foxp2_row["expr_bin"] = "anchor"
    selected_df = pd.concat([selected_df, foxp2_row], axis=0, ignore_index=True)

selected_genes = selected_df["gene"].tolist()
selected_df.to_csv(OUT_DIR / "lrt_unbiased_selected_genes.csv", index=False)

print("Selected genes (independent of group/logFC):")
print(selected_df[["gene", "sparsity_bin", "expr_bin", "zero_frac", "mean_expr", "detected_n"]].to_string(index=False))
print(f"\nTotal selected genes: {len(selected_genes)}")

fit_cfg = FitConfig(method="lbfgs", verbose=False, n_runs=2)
glm_cfg = GLMConfig(offset=False)


def run_lrt_eval(design_df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for model_name in ["nb", "mzinb"]:
        result = fit_glm(
            adata=adata,
            genes=selected_genes,
            X_design=design_df,
            model=model_name,
            fit_cfg=fit_cfg,
            glm_cfg=glm_cfg,
            n_jobs=4,
            seed=42,
        )
        result.add_lrt("group")

        for gene in selected_genes:
            gene_lrt = result[gene].get("lrt")
            if gene_lrt is None or len(gene_lrt) == 0:
                continue
            term_row = gene_lrt[gene_lrt["term"] == "group"]
            if len(term_row) == 0:
                continue
            row = term_row.iloc[0]
            meta = selected_df[selected_df["gene"] == gene].iloc[0]
            rows.append(
                {
                    "dataset": label,
                    "model": model_name,
                    "gene": gene,
                    "sparsity_bin": meta["sparsity_bin"],
                    "expr_bin": meta["expr_bin"],
                    "zero_frac": float(meta["zero_frac"]),
                    "mean_expr": float(meta["mean_expr"]),
                    "detected_n": int(meta["detected_n"]),
                    "LR": float(row["LR"]),
                    "p_value": float(row["p_value"]),
                    "ll_full": float(row["ll_full"]),
                    "ll_reduced": float(row["ll_reduced"]),
                    "full_refit_better": bool(row.get("full_refit_better", False)),
                    "reduced_better_than_full": bool(row.get("reduced_better_than_full", False)),
                }
            )
    out = pd.DataFrame(rows)
    out["neg_log10p"] = -np.log10(out["p_value"].clip(1e-300))
    return out


observed_df = run_lrt_eval(X_design.copy(), label="observed")
X_perm = X_design.copy()
X_perm["group"] = RNG.permutation(X_perm["group"].to_numpy())
permuted_df = run_lrt_eval(X_perm, label="permuted")

all_df = pd.concat([observed_df, permuted_df], axis=0, ignore_index=True)
all_df.to_csv(OUT_DIR / "lrt_nb_vs_mzinb_unbiased.csv", index=False)

print("\nSaved:")
print(OUT_DIR / "lrt_unbiased_selected_genes.csv")
print(OUT_DIR / "lrt_nb_vs_mzinb_unbiased.csv")

summary_rows = []
for dataset in ["observed", "permuted"]:
    for model_name in ["nb", "mzinb"]:
        sub = all_df[(all_df["dataset"] == dataset) & (all_df["model"] == model_name)]
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "n_genes": len(sub),
                "sig_p05": int((sub["p_value"] < 0.05).sum()),
                "median_p": float(sub["p_value"].median()),
                "median_neg_log10p": float(sub["neg_log10p"].median()),
                "mean_neg_log10p": float(sub["neg_log10p"].mean()),
                "reduced_better_than_full": int(sub["reduced_better_than_full"].sum()),
            }
        )
summary_df = pd.DataFrame(summary_rows)

print("\nOverall summary:")
print(summary_df.to_string(index=False))

print("\nBy sparsity bin:")
for sparsity_bin in ["low", "mid", "high"]:
    print(f"\n  {sparsity_bin}")
    for dataset in ["observed", "permuted"]:
        for model_name in ["nb", "mzinb"]:
            sub = all_df[
                (all_df["dataset"] == dataset)
                & (all_df["model"] == model_name)
                & (all_df["sparsity_bin"] == sparsity_bin)
            ]
            if len(sub) == 0:
                continue
            print(
                f"    {dataset:8s} {model_name:5s} "
                f"sig={int((sub['p_value'] < 0.05).sum())}/{len(sub)} "
                f"median -log10p={sub['neg_log10p'].median():.3f}"
            )

foxp2_df = all_df[all_df["gene"] == "FOXP2"].copy()
if len(foxp2_df) > 0:
    print("\nFOXP2 anchor:")
    print(foxp2_df[["dataset", "model", "LR", "p_value", "neg_log10p", "ll_full", "ll_reduced"]].to_string(index=False))

wide = all_df.pivot(index=["dataset", "gene"], columns="model", values=["p_value", "neg_log10p", "LR"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()
wide["delta_neg_log10p_mz_minus_nb"] = wide["neg_log10p_mzinb"] - wide["neg_log10p_nb"]

print("\nPer-gene comparison (observed only):")
obs_wide = wide[wide["dataset"] == "observed"].copy()
obs_wide = obs_wide.merge(selected_df[["gene", "sparsity_bin", "expr_bin", "zero_frac", "mean_expr"]], on="gene", how="left")
print(
    obs_wide[
        [
            "gene",
            "sparsity_bin",
            "expr_bin",
            "zero_frac",
            "mean_expr",
            "LR_nb",
            "p_value_nb",
            "LR_mzinb",
            "p_value_mzinb",
            "delta_neg_log10p_mz_minus_nb",
        ]
    ]
    .sort_values(["sparsity_bin", "expr_bin", "gene"])
    .to_string(index=False)
)
