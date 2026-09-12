"""Permutation FPR: NB vs MZINB — 20 permutations, LRT on 'group' term."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from deg_zinb import fit_glm
from deg_zinb.torch_backend.fit import FitConfig
from deg_zinb.torch_backend.model import GLMConfig

# ── Config ────────────────────────────────────────────────────────────────────
N_PERMS   = 20
ALPHA     = 0.05
N_JOBS    = 4
N_RUNS    = 2
OUT_DIR   = Path("tests/Results")
GENE_FILE = OUT_DIR / "lrt_unbiased_selected_genes.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
adata    = anndata.read_h5ad("tests/Results/adata_sub.h5ad")
X_design = pd.read_csv("tests/Results/X_design.csv", index_col=0)
sel_df   = pd.read_csv(GENE_FILE)
genes    = sel_df["gene"].tolist()

# Observed LRT (reuse previous result if available, otherwise recompute)
obs_path = OUT_DIR / "lrt_nb_vs_mzinb_unbiased.csv"
obs_all  = pd.read_csv(obs_path)
obs_all  = obs_all[obs_all["dataset"] == "observed"].copy()

cfg  = FitConfig(method="lbfgs", verbose=False, n_runs=N_RUNS)
gcfg = GLMConfig(offset=False)

RNG  = np.random.default_rng(0)

# ── Utils ─────────────────────────────────────────────────────────────────────
def lrt_for_design(design_df, label, seed):
    rows = []
    for model_name in ["nb", "mzinb"]:
        res = fit_glm(
            adata=adata, genes=genes, X_design=design_df,
            model=model_name, fit_cfg=cfg, glm_cfg=gcfg,
            n_jobs=N_JOBS, seed=seed,
        )
        res.add_lrt("group")
        for g in genes:
            gene_lrt = res[g].get("lrt")
            if gene_lrt is None or len(gene_lrt) == 0:
                continue
            r = gene_lrt[gene_lrt["term"] == "group"]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append({
                "perm_id":   label,
                "model":     model_name,
                "gene":      g,
                "LR":        float(r["LR"]),
                "p_value":   float(r["p_value"]),
                "ll_full":   float(r["ll_full"]),
                "ll_reduced":float(r["ll_reduced"]),
                "reduced_better": bool(r.get("reduced_better_than_full", False)),
            })
    return pd.DataFrame(rows)

# ── Run permutations ──────────────────────────────────────────────────────────
perm_records = []
t0 = time.time()
for i in range(N_PERMS):
    Xp = X_design.copy()
    Xp["group"] = RNG.permutation(Xp["group"].to_numpy())
    seed_i = 1000 + i
    df_i = lrt_for_design(Xp, label=i, seed=seed_i)
    perm_records.append(df_i)
    elapsed = time.time() - t0
    avg = elapsed / (i + 1)
    remaining = avg * (N_PERMS - i - 1)
    print(f"  perm {i+1:2d}/{N_PERMS}  "
          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

perm_df = pd.concat(perm_records, axis=0, ignore_index=True)
perm_df.to_csv(OUT_DIR / "perm_lrt_raw.csv", index=False)
print(f"\nSaved perm raw → {OUT_DIR/'perm_lrt_raw.csv'}")

# ── Per-gene empirical FPR ────────────────────────────────────────────────────
fpr_rows = []
for model_name in ["nb", "mzinb"]:
    sub = perm_df[perm_df["model"] == model_name]
    for g in genes:
        g_sub = sub[sub["gene"] == g]
        n_sig  = int((g_sub["p_value"] < ALPHA).sum())
        n_perm = len(g_sub)
        fpr_rows.append({
            "model": model_name, "gene": g,
            "n_perm": n_perm, "n_sig": n_sig,
            "empirical_fpr": n_sig / n_perm if n_perm > 0 else float("nan"),
        })

fpr_df = pd.DataFrame(fpr_rows)
fpr_df = fpr_df.merge(sel_df[["gene","zero_frac","detected_n","sparsity_bin","expr_bin"]], on="gene")

# Merge observed LRT
for model_name in ["nb", "mzinb"]:
    obs_m = obs_all[obs_all["model"]==model_name][["gene","LR","p_value","neg_log10p"]].copy()
    obs_m.columns = ["gene", f"obs_LR_{model_name}", f"obs_p_{model_name}", f"obs_nlp_{model_name}"]
    fpr_df = fpr_df.merge(obs_m, on="gene", how="left")

fpr_df.to_csv(OUT_DIR / "perm_fpr_summary.csv", index=False)

# ── Print Results ─────────────────────────────────────────────────────────────
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_rows", 60)

print("\n" + "="*70)
print(f"Empirical FPR  ({N_PERMS} permutations, alpha={ALPHA})")
print("="*70)

nb_fpr  = fpr_df[fpr_df["model"]=="nb"][["gene","sparsity_bin","detected_n","zero_frac","empirical_fpr","obs_LR_nb","obs_p_nb"]].set_index("gene")
mz_fpr  = fpr_df[fpr_df["model"]=="mzinb"][["gene","empirical_fpr","obs_LR_mzinb","obs_p_mzinb"]].set_index("gene")
nb_fpr.columns  = ["bin","detected_n","zero_frac","nb_fpr","obs_LR_nb","obs_p_nb"]
mz_fpr.columns  = ["mz_fpr","obs_LR_mz","obs_p_mz"]
wide = nb_fpr.join(mz_fpr)
wide["delta_fpr"] = wide["mz_fpr"] - wide["nb_fpr"]
wide = wide.sort_values(["bin","zero_frac"])
print(wide.to_string())

print("\n" + "="*70)
print("Summary by filter")
print("="*70)
filter_specs = [
    ("all genes",            wide["gene"].notna()                                   if "gene" in wide.columns else (wide.index.notna())),
    ("detected_n >= 100",   wide["detected_n"] >= 100),
    ("zero_frac < 0.98",    wide["zero_frac"] < 0.98),
    ("detected_n>=100 & zf<0.98", (wide["detected_n"]>=100)&(wide["zero_frac"]<0.98)),
]

def summarise(w, mask, name):
    s = w[mask]
    print(f"\n  {name}  (n={len(s)})")
    for col, model_name in [("nb_fpr","NB"),("mz_fpr","MZINB")]:
        print(f"    {model_name:5s}  mean_empirical_FPR={s[col].mean():.3f}"
              f"  >10%: {(s[col]>0.1).sum()}/{len(s)}"
              f"  >20%: {(s[col]>0.2).sum()}/{len(s)}")
    print(f"    MZINB-NB delta_fpr: mean={s['delta_fpr'].mean():+.3f}"
          f"  max={s['delta_fpr'].max():+.3f}")

for name, mask in filter_specs:
    summarise(wide, mask, name)

print("\n" + "="*70)
print("Observed vs expected: genes where observed p<0.05 but high perm FPR")
print("="*70)
for model_name, col_obs, col_fpr in [("NB","obs_p_nb","nb_fpr"),("MZINB","obs_p_mz","mz_fpr")]:
    suspect = wide[(wide[col_obs]<0.05) & (wide[col_fpr]>0.1)]
    if len(suspect)==0:
        print(f"  {model_name}: no suspects")
    else:
        print(f"\n  {model_name} suspects (obs p<0.05 but perm FPR>10%):")
        print(suspect[[col_obs, col_fpr, "zero_frac","detected_n"]].to_string())
