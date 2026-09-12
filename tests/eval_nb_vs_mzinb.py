"""NB vs MZINB evaluation: signal/null genes, stratified by sparsity."""
import anndata, pandas as pd, numpy as np, scipy.sparse as sp, time, warnings
warnings.filterwarnings('ignore')

from deg_zinb import fit_glm
from deg_zinb.torch_backend.fit import FitConfig
from deg_zinb.torch_backend.model import GLMConfig

adata = anndata.read_h5ad("tests/Results/adata_sub.h5ad")
X_design = pd.read_csv("tests/Results/X_design.csv", index_col=0)

Y = adata.X
if sp.issparse(Y): Y = Y.toarray()

zero_frac  = (Y == 0).mean(axis=0)
mask_pert  = X_design['group'].values == 1.0
mask_ctrl  = X_design['group'].values == 0.0
mean_pert  = Y[mask_pert].mean(axis=0) + 1e-3
mean_ctrl  = Y[mask_ctrl].mean(axis=0) + 1e-3
log2fc     = np.log2(mean_pert / mean_ctrl)
gene_names = np.array(adata.var_names)

gene_df = pd.DataFrame({'gene': gene_names, 'log2fc': log2fc,
    'abs_log2fc': np.abs(log2fc), 'zero_frac': zero_frac,
    'mean_expr': Y.mean(axis=0)})
gene_df['sparsity_bin'] = pd.cut(gene_df['zero_frac'],
    bins=[-0.001, 0.70, 0.95, 1.001], labels=['low', 'mid', 'high'])
well_expr = gene_df['mean_expr'] > 0.05

signal_genes, null_genes = [], []
for sbin in ['low', 'mid', 'high']:
    sub = gene_df[(gene_df['sparsity_bin'] == sbin) & well_expr]
    signal_genes += list(sub.nlargest(10, 'abs_log2fc')['gene'])
    null_genes   += list(sub.nsmallest(10, 'abs_log2fc')['gene'])

all_genes  = signal_genes + null_genes
gene_label = {g: 'signal' for g in signal_genes}
gene_label.update({g: 'null' for g in null_genes})
bin_label  = gene_df.set_index('gene')['sparsity_bin'].to_dict()
log2fc_lab = gene_df.set_index('gene')['log2fc'].to_dict()
zf_lab     = gene_df.set_index('gene')['zero_frac'].to_dict()

print(f"Total genes: {len(all_genes)} ({len(signal_genes)} signal, {len(null_genes)} null)")

cfg  = FitConfig(method='lbfgs', verbose=False, n_runs=3)
gcfg = GLMConfig(offset=False)
records = []

for model_name in ['nb', 'mzinb']:
    t0 = time.time()
    res = fit_glm(adata=adata, genes=all_genes, X_design=X_design,
                  model=model_name, fit_cfg=cfg, glm_cfg=gcfg, n_jobs=4, seed=42)
    res.add_wald()
    elapsed = time.time() - t0
    print(f"{model_name}: {len(res)} genes fit in {elapsed:.1f}s")

    for g in all_genes:
        row = res.get(g)
        if row is None: continue
        wald_df = row.get('wald')
        if wald_df is None: continue
        best_loss = row.get('best_loss', float('nan'))

        terms = list(wald_df['term'].values)
        group_terms = [t for t in terms if t.endswith(':group') or t == 'group']
        if not group_terms:
            print(f"  [WARN] no group term for {g} ({model_name}): {terms}")
            continue
        r = wald_df[wald_df['term'] == group_terms[0]].iloc[0]

        coef_val = r.get('Estimate', r.get('coef', r.get('coef_mean', float('nan'))))
        se_val   = r.get('Std. Error', r.get('se', float('nan')))
        z_val    = r.get('z value', r.get('z', float('nan')))
        p_val    = r.get('Pr(>|z|)', r.get('p_value', float('nan')))

        records.append({'model': model_name, 'gene': g,
            'set': gene_label[g], 'sparsity_bin': str(bin_label[g]),
            'log2fc_raw': float(log2fc_lab[g]), 'zero_frac': float(zf_lab[g]),
            'coef': float(coef_val), 'se': float(se_val), 'z': float(z_val),
            'p_value': float(p_val), 'best_loss': float(best_loss)})

df = pd.DataFrame(records)
out_path = "tests/Results/nb_vs_mzinb_eval.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}  ({len(df)} rows)")

if len(df) == 0:
    import sys; sys.exit(1)

df['neg_log10p'] = -np.log10(df['p_value'].clip(1e-300))

nb = df[df['model']=='nb'].set_index('gene')[['set','sparsity_bin','log2fc_raw','zero_frac','coef','se','z','p_value','neg_log10p','best_loss']].copy()
mz = df[df['model']=='mzinb'].set_index('gene')[['coef','se','z','p_value','neg_log10p','best_loss']].copy()
nb.columns = ['set','sparsity_bin','log2fc_raw','zero_frac','nb_coef','nb_se','nb_z','nb_p','nb_nlp','nb_loss']
mz.columns = ['mz_coef','mz_se','mz_z','mz_p','mz_nlp','mz_loss']
wide = nb.join(mz)
wide['delta_nlp'] = wide['mz_nlp'] - wide['nb_nlp']

pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_rows', 60)

print("\n" + "="*65)
print("SIGNAL genes -- NB vs MZINB (Wald, term=*:group)")
print("="*65)
sig = wide[wide['set']=='signal']
for sbin in ['low','mid','high']:
    s = sig[sig['sparsity_bin']==sbin]
    if len(s)==0: continue
    print(f"\n  bin={sbin} (n={len(s)}, zf=[{s['zero_frac'].min():.2f},{s['zero_frac'].max():.2f}], |log2FC|=[{s['log2fc_raw'].abs().min():.3f},{s['log2fc_raw'].abs().max():.3f}])")
    print(f"    NB    p<0.05: {(s['nb_p']<0.05).sum():2d}/{len(s)}  median -log10p={s['nb_nlp'].median():.2f}")
    print(f"    MZINB p<0.05: {(s['mz_p']<0.05).sum():2d}/{len(s)}  median -log10p={s['mz_nlp'].median():.2f}")
    print(f"    Coef corr: r={s['nb_coef'].corr(s['mz_coef']):.3f}")
    print(f"    Mean delta_nlp (MZINB-NB): {s['delta_nlp'].mean():+.3f}  (MZINB wins: {(s['delta_nlp']>0.1).sum()}, NB wins: {(s['delta_nlp']<-0.1).sum()})")

print("\n" + "="*65)
print("NULL genes -- calibration (FP rate, expect ~0.05 with n=10)")
print("="*65)
null = wide[wide['set']=='null']
for sbin in ['low','mid','high']:
    s = null[null['sparsity_bin']==sbin]
    if len(s)==0: continue
    print(f"\n  bin={sbin} (n={len(s)})")
    print(f"    NB    p<0.05: {(s['nb_p']<0.05).sum()}/{len(s)}  median p={s['nb_p'].median():.3f}")
    print(f"    MZINB p<0.05: {(s['mz_p']<0.05).sum()}/{len(s)}  median p={s['mz_p'].median():.3f}")

print("\n" + "="*65)
print("Per-gene detail -- SIGNAL")
print("="*65)
print(sig[['sparsity_bin','log2fc_raw','zero_frac','nb_coef','nb_p','nb_nlp','mz_coef','mz_p','mz_nlp','delta_nlp']
     ].sort_values(['sparsity_bin','log2fc_raw']).to_string())
