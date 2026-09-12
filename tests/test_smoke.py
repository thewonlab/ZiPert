from __future__ import annotations

from pathlib import Path
import sys

import anndata as ad
import pandas as pd
import torch
import math


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deg_zinb.api import cluster_genes_by_sparsity, fit_czinb_gene_sets, fit_czinb_glm, fit_glm
import deg_zinb.api as api_module
from deg_zinb.torch_backend.fit import FitConfig
from deg_zinb.torch_backend.model import GLMConfig


def _load_test_inputs():
    adata = ad.read_h5ad(ROOT / "tests" / "Results" / "adata_sub.h5ad")
    x_design = pd.read_csv(ROOT / "tests" / "Results" / "X_design.csv")
    gene = "ENSG00000238009"
    return adata, x_design, gene


def _fit_single_gene(model_name: str):
    adata, x_design, gene = _load_test_inputs()
    fit_cfg = FitConfig(method="lbfgs", max_iter=50, rtol=1e-6, n_runs=1, warm_start=True)
    result = fit_glm(
        adata=adata,
        genes=[gene],
        X_design=x_design,
        model=model_name,
        fit_cfg=fit_cfg,
        glm_cfg=GLMConfig(ridge=0.0),
        device="cpu",
        seed=1,
        dtype=torch.float64,
        n_jobs=1,
    )
    return result, gene


def _fit_czinb_multi_gene():
    adata, x_design, gene = _load_test_inputs()
    genes = [gene, "MIR1302-2HG", "LINC01409"]
    fit_cfg = FitConfig(method="lbfgs", max_iter=30, rtol=1e-6, n_runs=1, warm_start=True)
    model = fit_czinb_glm(
        adata=adata,
        genes=genes,
        X_design=x_design,
        fit_cfg=fit_cfg,
        glm_cfg=GLMConfig(ridge=0.0),
        device="cpu",
        seed=1,
        dtype=torch.float64,
    )
    return model, genes, x_design.shape[1]


def test_import_and_single_gene_fit_smoke():
    result, gene = _fit_single_gene("nb")

    assert gene in result
    assert result[gene]["model_kind"] == "nb"
    assert set(result[gene]["state_dict"]) == {"beta", "log_theta"}
    assert result[gene]["neg_log_likelihood"] > 0


def test_wald_and_lrt_smoke():
    result, gene = _fit_single_gene("nb")

    result.add_wald()
    assert result.wald_df is not None
    assert "count:group" in set(result.wald_df["term"])

    result.add_lrt("group")
    assert result.lrt_df is not None
    row = result.lrt_df.iloc[0]
    assert row["term"] == "group"
    assert row["LR"] >= 0.0
    assert 0.0 <= row["p_value"] <= 1.0


def test_zinb_and_mzinb_smoke():
    for model_name in ("zinb", "mzinb"):
        result, gene = _fit_single_gene(model_name)

        assert gene in result
        assert result[gene]["model_kind"] == model_name
        assert set(result[gene]["state_dict"]) == {"beta", "gamma", "log_theta"}
        assert result[gene]["neg_log_likelihood"] > 0

        result.add_wald()
        assert result.wald_df is not None
        assert "count:group" in set(result.wald_df["term"])
        assert "zero:group" in set(result.wald_df["term"])

        result.add_lrt("group")
        row = result.lrt_df.iloc[0]
        assert row["term"] == "group"
        assert row["LR"] >= 0.0
        assert 0.0 <= row["p_value"] <= 1.0


def test_czinb_fit_smoke():
    model, genes, p_mean = _fit_czinb_multi_gene()

    assert model.beta.shape == (len(genes), p_mean)
    assert model.log_theta.shape == (len(genes),)
    assert model.gamma.shape == (p_mean,)
    assert torch.isfinite(model.beta).all()
    assert torch.isfinite(model.log_theta).all()
    assert torch.isfinite(model.gamma).all()


def test_czinb_joint_likelihood_matches_shared_dropout_definition():
    from deg_zinb.torch_backend.model import CZINBGLM, GLMConfig, nb_log_prob

    Y = torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    X = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    model = CZINBGLM(n_genes=2, p_mean=1, cfg=GLMConfig(ridge=0.0)).to(dtype=torch.float64)

    with torch.no_grad():
        model.beta.copy_(torch.tensor([[math.log(1.5)], [math.log(2.0)]], dtype=torch.float64))
        model.gamma.copy_(torch.tensor([math.log(0.25 / 0.75)], dtype=torch.float64))
        model.log_theta.copy_(torch.log(torch.tensor([3.0, 4.0], dtype=torch.float64)))

    obs_nll = model.neg_log_lik_obs(Y, X)

    mu = model.forward_mu(X, None)
    pi = model.forward_pi(X).reshape(-1)
    theta = torch.exp(model.log_theta).view(1, -1)
    nb_ll = nb_log_prob(Y, mu, theta)
    nb_ll_sum = nb_ll.sum(dim=1)

    expected_all_zero = -torch.logaddexp(
        torch.log(pi[0]),
        torch.log1p(-pi[0]) + nb_ll_sum[0],
    )
    expected_mixed = -(torch.log1p(-pi[1]) + nb_ll_sum[1])

    assert torch.allclose(obs_nll[0], expected_all_zero)
    assert torch.allclose(obs_nll[1], expected_mixed)


def test_czinb_gene_sets_smoke():
    adata, x_design, gene = _load_test_inputs()
    gene_sets = {
        "housekeeping_like": [gene, "MIR1302-2HG"],
        "sparse_like": ["LINC01409", "DOES_NOT_EXIST"],
    }

    result = fit_czinb_gene_sets(
        adata=adata,
        gene_sets=gene_sets,
        X_design=x_design,
        fit_cfg=FitConfig(method="lbfgs", max_iter=20, rtol=1e-6, n_runs=1, warm_start=True),
        glm_cfg=GLMConfig(ridge=0.0),
        device="cpu",
        seed=1,
        dtype=torch.float64,
        min_genes=1,
        drop_missing=True,
    )

    assert set(result.keys()) == {"housekeeping_like", "sparse_like"}
    assert result.gene_sets["housekeeping_like"] == [gene, "MIR1302-2HG"]
    assert result.gene_sets["sparse_like"] == ["LINC01409"]
    assert result.missing_genes["sparse_like"] == ["DOES_NOT_EXIST"]
    assert result["housekeeping_like"].beta.shape[0] == 2
    assert result["sparse_like"].beta.shape[0] == 1


def test_cluster_genes_by_sparsity_respects_requested_number_of_sets():
    adata, _, gene = _load_test_inputs()
    genes = [gene, "MIR1302-2HG", "LINC01409", "ENSG00000239945", "ENSG00000241860"]

    gene_sets = cluster_genes_by_sparsity(
        adata,
        genes=genes,
        n_gene_sets=2,
        set_prefix="auto",
    )

    assert set(gene_sets.keys()) == {"auto_1", "auto_2"}
    assigned = sorted(g for members in gene_sets.values() for g in members)
    assert assigned == sorted(genes)


def test_fit_czinb_gene_sets_auto_clusters_when_sets_not_provided():
    adata, x_design, gene = _load_test_inputs()
    genes = [gene, "MIR1302-2HG", "LINC01409", "ENSG00000239945"]

    result = fit_czinb_gene_sets(
        adata=adata,
        genes=genes,
        gene_sets=None,
        X_design=x_design,
        fit_cfg=FitConfig(method="lbfgs", max_iter=15, rtol=1e-6, n_runs=1, warm_start=True),
        glm_cfg=GLMConfig(ridge=0.0),
        device="cpu",
        seed=1,
        dtype=torch.float64,
        n_gene_sets=2,
        min_genes=1,
    )

    assert len(result) == 2
    assigned = sorted(g for members in result.gene_sets.values() for g in members)
    assert assigned == sorted(genes)


def test_failed_gene_is_excluded_from_followup_inference(monkeypatch):
    adata, x_design, gene = _load_test_inputs()
    genes = [gene, "MIR1302-2HG"]
    original_fit_single_gene_job = api_module._fit_single_gene_job

    def flaky_fit_single_gene_job(args):
        if args[0] == "MIR1302-2HG":
            raise RuntimeError("synthetic failure")
        return original_fit_single_gene_job(args)

    monkeypatch.setattr(api_module, "_fit_single_gene_job", flaky_fit_single_gene_job)

    result = fit_glm(
        adata=adata,
        genes=genes,
        X_design=x_design,
        model="nb",
        fit_cfg=FitConfig(method="lbfgs", max_iter=30, rtol=1e-6, n_runs=1, warm_start=True),
        glm_cfg=GLMConfig(ridge=0.0),
        device="cpu",
        seed=1,
        dtype=torch.float64,
        n_jobs=1,
    )

    assert result.genes == [gene]
    assert set(result.keys()) == {gene}

    result.add_wald()
    assert set(result.wald_df["gene"]) == {gene}

    result.add_lrt("group")
    assert set(result.lrt_df["gene"]) == {gene}
