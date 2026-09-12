from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import copy
import math
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count
from typing import Optional, Union, List, Mapping, Sequence

from .torch_backend.model import NBGLM, ZINBGLM, MZINBGLM, CZINBGLM, GLMConfig
from .torch_backend.fit import fit_model, FitConfig
from torch.func import functional_call
from .io import anndata_to_tensors, anndata_to_matrix_tensors
from .inference import (
    compute_hessian_cov,
    compute_sandwich_cov_fast,
    attach_summary,
    as_list,
    lrt_test,
    compute_ll_from_loss,
)

from dataclasses import dataclass, field

@dataclass
class GLMFitResult:
    fit_results: dict
    adata: object
    genes: list[str]
    X_design: object
    model: str
    fit_cfg: object
    glm_cfg: object
    offset_key: Optional[str]
    device: str
    seed: Optional[int]
    deterministic: bool
    dtype: torch.dtype

    wald_df: Optional[pd.DataFrame] = None
    lrt_df: Optional[pd.DataFrame] = None
    
    def __getitem__(self, gene: str):
        return self.fit_results[gene]

    def __contains__(self, gene: str):
        return gene in self.fit_results

    def __iter__(self):
        return iter(self.fit_results)

    def __len__(self):
        return len(self.fit_results)

    def keys(self):
        return self.fit_results.keys()

    def values(self):
        return self.fit_results.values()

    def items(self):
        return self.fit_results.items()

    def get(self, gene: str, default=None):
        return self.fit_results.get(gene, default)

    def __repr__(self):
        return f"GLMFitResult(n_genes={len(self.fit_results)}, model={self.model})"
    
    def _get_coef_names_and_matrix(self):
        if hasattr(self.X_design, "columns"):
            coef_names = list(self.X_design.columns)
            X_design_np = self.X_design.to_numpy()
        else:
            X_design_np = self.X_design
            coef_names = [f"x{i}" for i in range(X_design_np.shape[1])]
        return coef_names, X_design_np
    
    def add_wald(
        self,
        hess_jitter: float = 1e-6,
        hess_use_pinv: bool = True,
        robust: bool = False,
    ):
        coef_names, X_design_np = self._get_coef_names_and_matrix()
        rows = []

        for g in self.genes:
            res = self.fit_results[g]

            y_t, X_t, off_t = anndata_to_tensors(
                self.adata, g, X_design_np, offset_key=self.offset_key
            )
            y_t = y_t.to(device=self.device, dtype=self.dtype)
            X_t = X_t.to(device=self.device, dtype=self.dtype)
            if off_t is not None:
                off_t = off_t.to(device=self.device, dtype=self.dtype)

            m = _rebuild_model_from_fitres(
                res, glm_cfg=self.glm_cfg, device=self.device, dtype=self.dtype
            )
            loss_fn = _make_loss_fn(res["model_kind"], m, y_t, X_t, off_t)

            if robust:
                loss_obs_builder = _make_loss_obs_builder(res["model_kind"], m, y_t, X_t, off_t)
                compute_sandwich_cov_fast(
                    m,
                    loss_fn,
                    loss_obs_builder,
                    device=self.device,
                    jitter=hess_jitter,
                    use_pinv=hess_use_pinv,
                )
            else:
                compute_hessian_cov(
                    m, loss_fn,
                    device=self.device,
                    jitter=hess_jitter,
                    use_pinv=hess_use_pinv
                )

            attach_summary(m, coef_names=coef_names, model_kind=res["model_kind"])

            summ = m.summary() if callable(m.summary) else m.summary
            summ = summ.copy() if hasattr(summ, "copy") else summ

            if summ.index.name == "term" or "term" not in summ.columns:
                summ = summ.reset_index()

            param_df = _extract_wald_param_df(m, coef_names, res["model_kind"])

            if len(param_df) > 0:
                summ = summ.merge(param_df, on="term", how="left")

            for col in ["coef_mean", "coef_zero", "log_theta", "theta"]:
                if col not in summ.columns:
                    summ[col] = np.nan

            summ["gene"] = g
            summ["cov_type"] = "sandwich" if robust else "hessian"

            self.fit_results[g]["wald"] = summ.copy()
            rows.append(summ)

        self.wald_df = pd.concat(rows, axis=0, ignore_index=True)
        return self

    def add_lrt(
        self,
        lrt_target: Union[str, List[str]],
    ):
        coef_names, X_design_np = self._get_coef_names_and_matrix()
        lrt_terms = as_list(lrt_target)
        term_to_idx = {name: i for i, name in enumerate(coef_names)}

        rows = []

        for gi, g in enumerate(self.genes):
            res = self.fit_results[g]

            y_t, X_t, off_t = anndata_to_tensors(
                self.adata, g, X_design_np, offset_key=self.offset_key
            )
            y_t = y_t.to(device=self.device, dtype=self.dtype)
            X_t = X_t.to(device=self.device, dtype=self.dtype)
            if off_t is not None:
                off_t = off_t.to(device=self.device, dtype=self.dtype)

            lrt_fit_cfg = copy.deepcopy(self.fit_cfg)
            lrt_fit_cfg.warm_start = False

            m_full_cached = _rebuild_model_from_fitres(
                res, glm_cfg=self.glm_cfg, device=self.device, dtype=self.dtype
            )
            loss_fn_full_cached = _make_loss_fn(res["model_kind"], m_full_cached, y_t, X_t, off_t)
            full_loss_cached = float(loss_fn_full_cached().detach().cpu().item())
            ll_full_cached = compute_ll_from_loss(full_loss_cached)

            def model_builder_full():
                return _fit_one_model(
                    res["model_kind"],
                    p=X_t.shape[1],
                    glm_cfg=self.glm_cfg,
                    device=self.device,
                )

            def loss_builder_full(m_full):
                return _make_loss_fn(res["model_kind"], m_full, y_t, X_t, off_t)

            full_seed = None if self.seed is None else (self.seed + gi * 1000)
            m_full_refit = _fit_with_multirun(
                model_builder_full,
                loss_builder_full,
                lrt_fit_cfg,
                self.device,
                full_seed,
                self.deterministic,
                self.dtype,
                y_t,
                X_t,
                off_t,
            )
            loss_fn_full_refit = loss_builder_full(m_full_refit)
            full_loss_refit = float(loss_fn_full_refit().detach().cpu().item())
            ll_full_refit = compute_ll_from_loss(full_loss_refit)

            if ll_full_cached >= ll_full_refit:
                m_full = m_full_cached
                ll_full = ll_full_cached
            else:
                m_full = m_full_refit
                ll_full = ll_full_refit

            df_lrt = 1 if res["model_kind"] == "nb" else 2

            for ti, term in enumerate(lrt_terms):
                drop_idx = term_to_idx[term]
                keep_mask = np.ones(len(coef_names), dtype=bool)
                keep_mask[drop_idx] = False
                keep_mask_t = torch.as_tensor(keep_mask, device=m_full.beta.device, dtype=torch.bool)   

                X_red = X_t[:, keep_mask_t]

                def model_builder_red():
                    m_red = _fit_one_model(
                        res["model_kind"],
                        p=X_red.shape[1],
                        glm_cfg=self.glm_cfg,
                        device=self.device,
                    )
                    m_red = _warm_start_reduced_from_full(
                        m_red=m_red,
                        m_full=m_full,
                        keep_mask=keep_mask,
                        model_kind=res["model_kind"],
                    )
                    return m_red

                def loss_builder_red(m_red):
                    return _make_loss_fn(res["model_kind"], m_red, y_t, X_red, off_t)

                term_seed = None if self.seed is None else (self.seed + gi * 1000 + ti + 1)

                m_red = _fit_with_multirun(
                    model_builder_red,
                    loss_builder_red,
                    lrt_fit_cfg,
                    self.device,
                    term_seed,
                    self.deterministic,
                    self.dtype,
                    y_t,
                    X_red,
                    off_t,
                )

                loss_fn_red = loss_builder_red(m_red)
                red_loss = float(loss_fn_red().detach().cpu().item())
                ll_red = compute_ll_from_loss(red_loss)

                LR, pval = lrt_test(ll_full, ll_red, df_lrt)

                rows.append({
                    "gene": g,
                    "term": term,
                    "LR": LR,
                    "df": df_lrt,
                    "p_value": pval,
                    "ll_full": ll_full,
                    "ll_reduced": ll_red,
                    "full_refit_better": ll_full_refit > ll_full_cached,
                    "reduced_better_than_full": ll_red > ll_full,
                })

        self.lrt_df = pd.DataFrame(rows)

        for g in self.genes:
            self.fit_results[g]["lrt"] = self.lrt_df[self.lrt_df["gene"] == g].copy()

        return self


@dataclass
class CZINBGeneSetFitResult:
    models: dict[str, CZINBGLM]
    gene_sets: dict[str, list[str]]
    missing_genes: dict[str, list[str]]

    def __getitem__(self, gene_set_name: str):
        return self.models[gene_set_name]

    def __contains__(self, gene_set_name: str):
        return gene_set_name in self.models

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def keys(self):
        return self.models.keys()

    def values(self):
        return self.models.values()

    def items(self):
        return self.models.items()

    def get(self, gene_set_name: str, default=None):
        return self.models.get(gene_set_name, default)

    def __repr__(self):
        return f"CZINBGeneSetFitResult(n_sets={len(self.models)})"


def _as_gene_list(genes) -> list[str]:
    return [str(g) for g in genes]


def _get_candidate_genes_for_sets(adata, gene_sets=None, genes=None) -> list[str]:
    if isinstance(gene_sets, Mapping):
        raise TypeError("gene_sets is already a mapping; candidate genes are not needed.")
    if gene_sets is not None:
        return _as_gene_list(gene_sets)
    if genes is not None:
        return _as_gene_list(genes)
    return _as_gene_list(adata.var_names)


def _compute_gene_sparsity(adata, genes: Sequence[str]) -> np.ndarray:
    Y = adata[:, list(genes)].X
    if hasattr(Y, "toarray"):
        Y = Y.toarray()
    else:
        Y = np.asarray(Y)
    return np.mean(Y == 0, axis=0, dtype=np.float64)


def _kmeans_1d(values: np.ndarray, n_clusters: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.shape[0]
    if n_clusters < 1 or n_clusters > n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")

    if n_clusters == 1:
        center = np.array([values.mean()], dtype=np.float64)
        labels = np.zeros(n, dtype=int)
        sse = float(np.square(values - center[0]).sum())
        return labels, center, sse

    quantiles = np.linspace(0.0, 1.0, num=n_clusters)
    centers = np.quantile(values, quantiles).astype(np.float64)

    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = values[mask].mean()

        new_centers.sort()
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    distances = np.abs(values[:, None] - centers[None, :])
    labels = np.argmin(distances, axis=1)
    sse = float(np.square(values - centers[labels]).sum())
    return labels, centers, sse


def _auto_select_n_gene_sets(
    sparsity: np.ndarray,
    *,
    min_sets: int = 2,
    max_sets: int = 4,
) -> int:
    n_genes = int(len(sparsity))
    if n_genes <= 1:
        return 1

    min_sets_eff = max(1, min(min_sets, n_genes))
    max_sets_eff = max(min_sets_eff, min(max_sets, n_genes))
    best_k = min_sets_eff
    best_score = float("inf")

    for k in range(min_sets_eff, max_sets_eff + 1):
        _, _, sse = _kmeans_1d(sparsity, k)
        sse = max(sse, 1e-12)
        bic_like = n_genes * np.log(sse / n_genes) + k * np.log(n_genes)
        if bic_like < best_score:
            best_score = bic_like
            best_k = k

    return best_k


def cluster_genes_by_sparsity(
    adata,
    *,
    genes=None,
    n_gene_sets: Optional[int] = None,
    min_gene_sets_auto: int = 2,
    max_gene_sets_auto: int = 4,
    set_prefix: str = "sparsity",
):
    candidate_genes = _get_candidate_genes_for_sets(adata, gene_sets=None, genes=genes)
    if len(candidate_genes) == 0:
        raise ValueError("No genes provided for sparsity clustering.")

    sparsity = _compute_gene_sparsity(adata, candidate_genes)
    if n_gene_sets is None:
        n_gene_sets = _auto_select_n_gene_sets(
            sparsity,
            min_sets=min_gene_sets_auto,
            max_sets=max_gene_sets_auto,
        )

    labels, centers, _ = _kmeans_1d(sparsity, int(n_gene_sets))
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}

    grouped = {f"{set_prefix}_{i + 1}": [] for i in range(len(order))}
    for gene, label in zip(candidate_genes, labels):
        grouped[f"{set_prefix}_{remap[int(label)] + 1}"].append(gene)

    grouped = {k: v for k, v in grouped.items() if len(v) > 0}
    return grouped

@dataclass
class GLMRunner:
    adata: object
    genes: list[str]
    X_design: object
    model: str = "nb"
    offset_key: Optional[str] = None
    fit_cfg: object = field(default_factory=FitConfig)
    glm_cfg: object = field(default_factory=GLMConfig)
    device: str = "cpu"
    seed: Optional[int] = None
    deterministic: bool = False
    dtype: torch.dtype = torch.float64
    n_jobs: int = 1
    
    def _get_coef_names_and_matrix(self):
        if hasattr(self.X_design, "columns"):
            coef_names = list(self.X_design.columns)
            X_design_np = self.X_design.to_numpy()
        else:
            X_design_np = self.X_design
            coef_names = [f"x{i}" for i in range(X_design_np.shape[1])]
        return coef_names, X_design_np
    
    def fit(self):
        coef_names, X_design_np = self._get_coef_names_and_matrix()
        kind = self.model.lower()

        if self.offset_key is not None:
            off_np = np.asarray(self.adata.obs[self.offset_key]).reshape(-1)
        else:
            off_np = None

        Y_all = self.adata[:, self.genes].X
        if hasattr(Y_all, "toarray"):
            Y_all = Y_all.toarray()
        else:
            Y_all = np.asarray(Y_all)

        jobs = []
        for i, g in enumerate(self.genes):
            y_np = Y_all[:, i]
            gene_seed = None if self.seed is None else (self.seed + i)

            jobs.append((
                g,
                y_np,
                X_design_np,
                off_np,
                coef_names,
                kind,
                self.fit_cfg,
                self.glm_cfg,
                self.device,
                gene_seed,
                self.deterministic,
                self.dtype,
            ))

        results = {}
        failed_genes = {}

        # Adaptive policy: avoid process overhead for single-gene fits,
        # and never use more workers than genes in this call.
        if self.n_jobs == -1:
            requested_jobs = cpu_count() or 1
        else:
            requested_jobs = int(self.n_jobs)

        if requested_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1 or -1, got {self.n_jobs}")

        n_jobs_eff = min(requested_jobs, max(1, len(jobs)))

        if n_jobs_eff == 1:
            for job in jobs:
                try:
                    out = _fit_single_gene_job(job)
                    results[out["gene"]] = out
                except Exception as e:
                    gene_name = job[0]
                    failed_genes[gene_name] = str(e)
                    print(f"[WARN] skipping failed gene: {gene_name} | {e}")
        else:
            with ProcessPoolExecutor(max_workers=n_jobs_eff) as ex:
                fut_to_gene = {ex.submit(_fit_single_gene_job, job): job[0] for job in jobs}
                for fut in as_completed(fut_to_gene):
                    gene_name = fut_to_gene[fut]
                    try:
                        out = fut.result()
                        results[out["gene"]] = out
                    except Exception as e:
                        failed_genes[gene_name] = str(e)
                        print(f"[WARN] skipping failed gene: {gene_name} | {e}")

        if failed_genes:
            print(f"[WARN] {len(failed_genes)} genes failed and were skipped.")
            
        successful_genes = [g for g in self.genes if g in results]

        return GLMFitResult(
            fit_results=results,
            adata=self.adata,
            genes=successful_genes,
            X_design=self.X_design,
            model=self.model,
            fit_cfg=self.fit_cfg,
            glm_cfg=self.glm_cfg,
            offset_key=self.offset_key,
            device=self.device,
            seed=self.seed,
            deterministic=self.deterministic,
            dtype=self.dtype,
        )
        
def _fit_one_model(kind: str, p: int, glm_cfg, device: str):
    kind = kind.lower()
    if kind == "nb":
        return NBGLM(p=p, cfg=glm_cfg).to(device)
    if kind == "zinb":
        return ZINBGLM(p_mean=p, cfg=glm_cfg).to(device)
    if kind == "mzinb":
        return MZINBGLM(p_mean=p, cfg=glm_cfg).to(device)
    raise ValueError("model must be 'nb', 'zinb', or 'mzinb'")

def _make_loss_fn(kind: str, m, y_t, X_t, off_t):
    kind = kind.lower()
    if kind == "nb":
        return lambda: m.neg_log_lik(y_t, X_t, log_offset=off_t)
    return lambda: m.neg_log_lik(y_t, X_t, Z=None, log_offset=off_t)

def _make_loss_obs_fn(kind: str, m, y_t, X_t, off_t):
    kind = kind.lower()
    if kind == "nb":
        return lambda: m.neg_log_lik_obs(y_t, X_t, log_offset=off_t)
    return lambda: m.neg_log_lik_obs(y_t, X_t, Z=None, log_offset=off_t)


def _make_loss_obs_builder(kind: str, m, y_t, X_t, off_t):
    kind = kind.lower()

    def builder(params_dict):
        if kind == "nb":
            return functional_call(
                m,
                params_dict,
                (y_t, X_t),
                {"log_offset": off_t}
            )
        else:
            return functional_call(
                m,
                params_dict,
                (y_t, X_t),
                {"Z": None, "log_offset": off_t}
            )

    return builder


def _fit_with_multirun(
    model_builder,
    loss_builder,
    fit_cfg,
    device,
    seed,
    deterministic,
    dtype,
    y,
    X,
    log_offset,
):
    best_model = None
    best_loss = float("inf")
    best_err = None
    n_runs = getattr(fit_cfg, "n_runs", 1)

    for i in range(n_runs):
        run_seed = None if seed is None else seed + i

        m = model_builder()
        loss_fn = loss_builder(m)

        try:
            fit_model(
                m,
                loss_fn,
                fit_cfg,
                device=device,
                y=y,
                X=X,
                log_offset=log_offset,
                seed=run_seed,
                deterministic=deterministic,
                dtype=dtype,
            )

            with torch.no_grad():
                loss_val = float(loss_fn().detach().cpu())

            if not math.isfinite(loss_val):
                msg = f"non-finite final loss at run {i}, seed={run_seed}: {loss_val}"
                print(f"[WARN] {msg}")
                best_err = msg
                continue

            if loss_val < best_loss:
                best_loss = loss_val
                best_model = m

        except Exception as e:
            msg = f"fit failed at run {i}, seed={run_seed}: {repr(e)}"
            print(f"[WARN] {msg}")
            best_err = msg
            continue

    if best_model is None:
        raise RuntimeError(
            f"All multirun fits failed or returned non-finite loss. "
            f"seed={seed}, n_runs={n_runs}, last_error={best_err}"
        )

    return best_model

def _fit_single_gene_job(args):
    (
        g,
        y_np,
        X_design_np,
        off_np,
        coef_names,
        kind,
        fit_cfg,
        glm_cfg,
        device,
        seed,
        deterministic,
        dtype,
    ) = args

    try:
        y_t = torch.as_tensor(y_np, device=device, dtype=dtype).reshape(-1)
        X_t = torch.as_tensor(X_design_np, device=device, dtype=dtype)
        off_t = None if off_np is None else torch.as_tensor(off_np, device=device, dtype=dtype).reshape(-1)

        # input validation
        if not torch.isfinite(y_t).all():
            raise RuntimeError(f"Non-finite y detected for gene={g}")
        if not torch.isfinite(X_t).all():
            raise RuntimeError(f"Non-finite X detected for gene={g}")
        if off_t is not None and not torch.isfinite(off_t).all():
            raise RuntimeError(f"Non-finite offset detected for gene={g}")
        if (y_t < 0).any():
            raise RuntimeError(f"Negative counts detected for gene={g}")

        def model_builder():
            return _fit_one_model(kind, p=X_t.shape[1], glm_cfg=glm_cfg, device=device)

        def loss_builder(m):
            return _make_loss_fn(kind, m, y_t, X_t, off_t)

        m = _fit_with_multirun(
            model_builder,
            loss_builder,
            fit_cfg,
            device,
            seed,
            deterministic,
            dtype,
            y_t,
            X_t,
            off_t,
        )

        loss_fn = loss_builder(m)
        return _pack_fitted_model(g, m, loss_fn, kind, coef_names)

    except Exception as e:
        debug_msg = (
            f"Failed gene job: gene={g}, seed={seed}, "
            f"y_min={float(torch.min(y_t).detach().cpu()) if 'y_t' in locals() else 'NA'}, "
            f"y_max={float(torch.max(y_t).detach().cpu()) if 'y_t' in locals() else 'NA'}, "
            f"y_zero_frac={float((y_t == 0).to(torch.float64).mean().detach().cpu()) if 'y_t' in locals() else 'NA'}, "
            f"X_min={float(torch.min(X_t).detach().cpu()) if 'X_t' in locals() else 'NA'}, "
            f"X_max={float(torch.max(X_t).detach().cpu()) if 'X_t' in locals() else 'NA'}, "
            f"off_min={float(torch.min(off_t).detach().cpu()) if ('off_t' in locals() and off_t is not None) else 'NA'}, "
            f"off_max={float(torch.max(off_t).detach().cpu()) if ('off_t' in locals() and off_t is not None) else 'NA'}"
        )
        raise RuntimeError(f"{debug_msg}") from e


def _pack_fitted_model(g, m, loss_fn, kind, coef_names):
    return {
        "gene": g,
        "model_kind": kind,
        "coef_names": list(coef_names),
        "state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()},
        "neg_log_likelihood": float(loss_fn().detach().cpu().item()),
        "wald": None,
        "lrt": None,
    }

def _rebuild_model_from_fitres(res, glm_cfg, device="cpu", dtype=torch.float64):
    kind = res["model_kind"]
    coef_names = res["coef_names"]
    p = len(coef_names)

    m = _fit_one_model(kind, p=p, glm_cfg=glm_cfg, device=device)
    m = m.to(device=device, dtype=dtype)
    m.load_state_dict(res["state_dict"])
    return m

def _extract_wald_param_df(m, coef_names, model_kind):
    rows = []

    # mean beta
    if hasattr(m, "beta") and m.beta is not None:
        beta = m.beta.detach().cpu().reshape(-1).numpy()
        for nm, val in zip(coef_names, beta):
            rows.append({
                "term": f"count:{nm}",
                "coef_mean": float(val),
            })

    # zero gamma
    if model_kind in ("zinb", "mzinb"):
        if hasattr(m, "gamma") and m.gamma is not None:
            gamma = m.gamma.detach().cpu().reshape(-1).numpy()
            for nm, val in zip(coef_names, gamma):
                rows.append({
                    "term": f"zero:{nm}",
                    "coef_zero": float(val),
                })

    # theta / log_theta
    if hasattr(m, "log_theta") and m.log_theta is not None:
        log_theta = float(m.log_theta.detach().cpu().reshape(-1)[0].item())
        theta = float(torch.exp(m.log_theta.detach().cpu().reshape(-1)[0]).item())
        rows.append({
            "term": "log(theta)",
            "log_theta": log_theta,
            "theta": theta,
        })

    param_df = pd.DataFrame(rows)

    if len(param_df) == 0:
        return param_df

    param_df = param_df.groupby("term", as_index=False).first()
    return param_df


def _warm_start_reduced_from_full(m_red, m_full, keep_mask, model_kind):
    keep_mask_t = torch.as_tensor(keep_mask, device=m_full.beta.device, dtype=torch.bool)

    if hasattr(m_full, "beta") and hasattr(m_red, "beta"):
        with torch.no_grad():
            m_red.beta.copy_(m_full.beta.detach()[keep_mask_t])

    if model_kind in ("zinb", "mzinb"):
        if hasattr(m_full, "gamma") and hasattr(m_red, "gamma"):
            with torch.no_grad():
                m_red.gamma.copy_(m_full.gamma.detach()[keep_mask_t])

    if hasattr(m_full, "log_theta") and hasattr(m_red, "log_theta"):
        with torch.no_grad():
            m_red.log_theta.copy_(m_full.log_theta.detach())

    return m_red

def fit_glm(
    adata,
    genes,
    X_design,
    model="nb",
    offset_key=None,
    fit_cfg=None,
    glm_cfg=None,
    device="cpu",
    *,
    seed=None,
    deterministic=False,
    dtype=torch.float64,
    n_jobs=1,
):
    if fit_cfg is None:
        fit_cfg = FitConfig()
    if glm_cfg is None:
        glm_cfg = GLMConfig()
    runner = GLMRunner(
        adata=adata,
        genes=genes,
        X_design=X_design,
        model=model,
        offset_key=offset_key,
        fit_cfg=fit_cfg,
        glm_cfg=glm_cfg,
        device=device,
        seed=seed,
        deterministic=deterministic,
        dtype=dtype,
        n_jobs=n_jobs,
    )
    return runner.fit()


def fit_czinb_glm(
    adata,
    genes,
    X_design,
    Z_design=None,
    offset_key=None,
    fit_cfg=None,
    glm_cfg=None,
    device="cpu",
    seed=None,
    deterministic=False,
    dtype=torch.float64,
    init_model: Optional[CZINBGLM] = None,
    train_count: bool = True,
    train_zero: bool = True,
    train_theta: bool = True,
):
    if fit_cfg is None:
        fit_cfg = FitConfig()
    if glm_cfg is None:
        glm_cfg = GLMConfig()
    Y, X, log_offset = anndata_to_matrix_tensors(adata, genes, X_design, offset_key)

    if Z_design is None:
        Z = X.clone()
    else:
        if hasattr(Z_design, "values"):
            Z_np = np.asarray(Z_design.values, dtype=np.float64)
        else:
            Z_np = np.asarray(Z_design, dtype=np.float64)
        Z = torch.tensor(Z_np, dtype=torch.float64)

    Y = Y.to(device=device, dtype=dtype)
    X = X.to(device=device, dtype=dtype)
    Z = Z.to(device=device, dtype=dtype)
    if log_offset is not None:
        log_offset = log_offset.to(device=device, dtype=dtype)

    def model_builder():
        m = CZINBGLM(
            n_genes=Y.shape[1],
            p_mean=X.shape[1],
            p_zi=Z.shape[1],
            cfg=glm_cfg,
        ).to(device=device, dtype=dtype)

        if init_model is not None:
            # Safe warm-start for possibly different reduced/full count design dimensions.
            with torch.no_grad():
                if hasattr(init_model, "gamma") and init_model.gamma.shape == m.gamma.shape:
                    m.gamma.copy_(init_model.gamma)
                if hasattr(init_model, "log_theta") and init_model.log_theta.shape == m.log_theta.shape:
                    m.log_theta.copy_(init_model.log_theta)

                # beta can have different p_mean between full/reduced models.
                p_common = min(init_model.beta.shape[1], m.beta.shape[1])
                if p_common > 0:
                    m.beta[:, :p_common].copy_(init_model.beta[:, :p_common])

        m.beta.requires_grad_(bool(train_count))
        m.gamma.requires_grad_(bool(train_zero))
        m.log_theta.requires_grad_(bool(train_theta))
        return m

    def loss_builder(m):
        return lambda: m.neg_log_lik(Y, X, Z=Z, log_offset=log_offset)

    if not any([train_count, train_zero, train_theta]):
        frozen_model = model_builder()
        return frozen_model

    model = _fit_with_multirun(
        model_builder,
        loss_builder,
        fit_cfg,
        device,
        seed,
        deterministic,
        dtype,
        Y,
        X,
        log_offset,
    )
    return model


def lrt_czinb(
    adata,
    genes,
    X_count_design,
    lrt_target: Union[str, List[str]],
    *,
    X_zero_design=None,
    offset_key=None,
    fit_cfg=None,
    glm_cfg=None,
    device="cpu",
    seed=None,
    deterministic=False,
    dtype=torch.float64,
    mode: str = "count_only",
):
    """
    LRT for CZINB with selectable reduced-model refit strategy.

    mode:
      - "count_only": keep zero model fixed (gamma frozen from full), refit count/theta only.
      - "full_refit": refit both count and zero parts in reduced model.
    """
    if fit_cfg is None:
        fit_cfg = FitConfig()
    if glm_cfg is None:
        glm_cfg = GLMConfig()

    if mode not in ("count_only", "full_refit"):
        raise ValueError(f"mode must be 'count_only' or 'full_refit', got {mode}")

    if not hasattr(X_count_design, "columns"):
        raise TypeError("X_count_design must be a pandas DataFrame with named columns.")

    x_full = X_count_design.copy()
    terms = as_list(lrt_target)
    missing_terms = [t for t in terms if t not in x_full.columns]
    if missing_terms:
        raise ValueError(f"lrt_target terms not in X_count_design: {missing_terms}")

    keep_cols = [c for c in x_full.columns if c not in terms]
    if len(keep_cols) == 0:
        raise ValueError("Reduced count design is empty after removing lrt_target terms.")

    x_red = x_full.loc[:, keep_cols]
    z_full = x_full.copy() if X_zero_design is None else X_zero_design.copy()
    if not hasattr(z_full, "columns"):
        raise TypeError("X_zero_design must be a pandas DataFrame with named columns.")

    z_keep_cols = [c for c in z_full.columns if c not in terms]
    if len(z_keep_cols) == 0:
        raise ValueError("Reduced zero design is empty after removing lrt_target terms.")

    z_red = z_full.loc[:, z_keep_cols]

    full_model = fit_czinb_glm(
        adata=adata,
        genes=genes,
        X_design=x_full,
        Z_design=z_full,
        offset_key=offset_key,
        fit_cfg=fit_cfg,
        glm_cfg=glm_cfg,
        device=device,
        seed=seed,
        deterministic=deterministic,
        dtype=dtype,
        train_count=True,
        train_zero=True,
        train_theta=True,
    )

    Y_full, X_full_t, off_t = anndata_to_matrix_tensors(adata, genes, x_full, offset_key)
    Y_full = Y_full.to(device=device, dtype=dtype)
    X_full_t = X_full_t.to(device=device, dtype=dtype)
    if hasattr(z_full, "values"):
        z_np = np.asarray(z_full.values, dtype=np.float64)
    else:
        z_np = np.asarray(z_full, dtype=np.float64)
    Z_full_t = torch.tensor(z_np, dtype=torch.float64).to(device=device, dtype=dtype)
    if off_t is not None:
        off_t = off_t.to(device=device, dtype=dtype)

    with torch.no_grad():
        full_loss = float(full_model.neg_log_lik(Y_full, X_full_t, Z=Z_full_t, log_offset=off_t).detach().cpu().item())
    ll_full = compute_ll_from_loss(full_loss)

    if mode == "count_only":
        red_model = fit_czinb_glm(
            adata=adata,
            genes=genes,
            X_design=x_red,
            Z_design=z_full,
            offset_key=offset_key,
            fit_cfg=fit_cfg,
            glm_cfg=glm_cfg,
            device=device,
            seed=seed,
            deterministic=deterministic,
            dtype=dtype,
            init_model=full_model,
            train_count=True,
            train_zero=False,
            train_theta=True,
        )
    else:
        red_model = fit_czinb_glm(
            adata=adata,
            genes=genes,
            X_design=x_red,
            Z_design=z_red,
            offset_key=offset_key,
            fit_cfg=fit_cfg,
            glm_cfg=glm_cfg,
            device=device,
            seed=seed,
            deterministic=deterministic,
            dtype=dtype,
            train_count=True,
            train_zero=True,
            train_theta=True,
        )

    Y_red, X_red_t, off_red_t = anndata_to_matrix_tensors(adata, genes, x_red, offset_key)
    Y_red = Y_red.to(device=device, dtype=dtype)
    X_red_t = X_red_t.to(device=device, dtype=dtype)
    if hasattr(z_red, "values"):
        z_red_np = np.asarray(z_red.values, dtype=np.float64)
    else:
        z_red_np = np.asarray(z_red, dtype=np.float64)
    Z_red_t = torch.tensor(z_red_np, dtype=torch.float64).to(device=device, dtype=dtype)
    if off_red_t is not None:
        off_red_t = off_red_t.to(device=device, dtype=dtype)

    with torch.no_grad():
        red_Z_t = Z_full_t if mode == "count_only" else Z_red_t
        red_loss = float(red_model.neg_log_lik(Y_red, X_red_t, Z=red_Z_t, log_offset=off_red_t).detach().cpu().item())
    ll_red = compute_ll_from_loss(red_loss)

    zero_terms_removed = len([t for t in terms if t in z_full.columns]) if mode == "full_refit" else 0
    df = int((len(terms) + zero_terms_removed) * len(genes))
    LR, pval = lrt_test(ll_full, ll_red, df)

    out = pd.DataFrame(
        [
            {
                "mode": mode,
                "n_genes": int(len(genes)),
                "dropped_terms": ",".join(terms),
                "df": int(df),
                "ll_full": float(ll_full),
                "ll_reduced": float(ll_red),
                "LR": float(LR),
                "p_value": float(pval),
            }
        ]
    )

    return {
        "table": out,
        "full_model": full_model,
        "reduced_model": red_model,
    }


def fit_czinb_gene_sets(
    adata,
    X_design,
    gene_sets: Optional[Union[Mapping[str, List[str]], Sequence[str]]] = None,
    genes=None,
    offset_key=None,
    fit_cfg=None,
    glm_cfg=None,
    device="cpu",
    seed=None,
    deterministic=False,
    dtype=torch.float64,
    *,
    n_gene_sets: Optional[int] = None,
    min_gene_sets_auto: int = 2,
    max_gene_sets_auto: int = 4,
    set_prefix: str = "sparsity",
    min_genes: int = 1,
    drop_missing: bool = True,
):
    if fit_cfg is None:
        fit_cfg = FitConfig()
    if glm_cfg is None:
        glm_cfg = GLMConfig()

    if min_genes < 1:
        raise ValueError(f"min_genes must be >= 1, got {min_genes}")

    if isinstance(gene_sets, Mapping):
        resolved_gene_sets = dict(gene_sets)
    else:
        resolved_gene_sets = cluster_genes_by_sparsity(
            adata,
            genes=_get_candidate_genes_for_sets(adata, gene_sets=gene_sets, genes=genes),
            n_gene_sets=n_gene_sets,
            min_gene_sets_auto=min_gene_sets_auto,
            max_gene_sets_auto=max_gene_sets_auto,
            set_prefix=set_prefix,
        )

    available_genes = set(map(str, adata.var_names))
    models = {}
    used_gene_sets = {}
    missing_by_set = {}

    for si, (set_name, genes_in_input) in enumerate(resolved_gene_sets.items()):
        if len(genes_in_input) == 0:
            raise ValueError(f"Gene set '{set_name}' is empty.")

        genes_in_set = [str(g) for g in genes_in_input]
        present = [g for g in genes_in_set if g in available_genes]
        missing = [g for g in genes_in_set if g not in available_genes]

        if missing and not drop_missing:
            raise ValueError(
                f"Gene set '{set_name}' contains genes absent from adata.var_names: {missing}"
            )

        if len(present) < min_genes:
            raise ValueError(
                f"Gene set '{set_name}' has {len(present)} usable genes after filtering; "
                f"requires at least {min_genes}."
            )

        set_seed = None if seed is None else (seed + si * 1000)
        model = fit_czinb_glm(
            adata=adata,
            genes=present,
            X_design=X_design,
            offset_key=offset_key,
            fit_cfg=fit_cfg,
            glm_cfg=glm_cfg,
            device=device,
            seed=set_seed,
            deterministic=deterministic,
            dtype=dtype,
        )
        models[set_name] = model
        used_gene_sets[set_name] = present
        missing_by_set[set_name] = missing

    return CZINBGeneSetFitResult(
        models=models,
        gene_sets=used_gene_sets,
        missing_genes=missing_by_set,
    )
