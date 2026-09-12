# inference.py
from __future__ import annotations

import types
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, jacrev

# ----------------------------
# Hessian-based inference
# ----------------------------

def _get_trainable_params(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

def _flatten_params(params):
    return torch.nn.utils.parameters_to_vector(params)

def _named_trainable_params(model: torch.nn.Module):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}

def _param_slices(model: torch.nn.Module):
    out = []
    start = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        out.append((name, slice(start, start + n), tuple(p.shape)))
        start += n
    return out

def _normal_pvals_from_z(z: np.ndarray) -> np.ndarray:
    zt = torch.tensor(np.abs(z) / np.sqrt(2.0), dtype=torch.float64)
    return torch.special.erfc(zt).cpu().numpy()


def _safe_z_and_p(theta_hat: np.ndarray, se: np.ndarray):
    z = np.divide(
        theta_hat,
        se,
        out=np.full_like(theta_hat, np.nan, dtype=np.float64),
        where=se > 0,
    )
    pvals = np.full_like(theta_hat, np.nan, dtype=np.float64)
    valid = np.isfinite(z)
    if np.any(valid):
        pvals[valid] = _normal_pvals_from_z(z[valid])
    return z, pvals

def _flatten_grads(grads, params):
    out = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p).reshape(-1))
        else:
            out.append(g.reshape(-1))
    return torch.cat(out)


def _named_trainable_params(model: torch.nn.Module):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def _flatten_grads(grads, params):
    out = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p).reshape(-1))
        else:
            out.append(g.reshape(-1))
    return torch.cat(out)


def score_obs_wrt_params_fast(model, loss_obs_builder):
    """
    loss_obs_builder(params_dict) -> tensor of shape (n_obs,)
    returns:
        score_mat: (n_obs, n_params)
    """
    params_dict = _named_trainable_params(model)
    jac = jacrev(loss_obs_builder)(params_dict)

    blocks = []
    for name, p in params_dict.items():
        j = jac[name].reshape(jac[name].shape[0], -1)
        blocks.append(j)

    score_mat = torch.cat(blocks, dim=1)
    return score_mat.detach().cpu().double()


def compute_sandwich_cov_fast(
    model: torch.nn.Module,
    loss_fn,
    loss_obs_builder,
    *,
    device: str = "cpu",
    jitter: float = 1e-6,
    use_pinv: bool = True,
):
    """
    Robust sandwich covariance:
      cov = H^{-1} S H^{-1}
    """
    model.to(device)

    with torch.enable_grad():
        params = _get_trainable_params(model)
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found.")

        loss = loss_fn()
        if loss.dim() != 0:
            raise ValueError("loss_fn must return a scalar tensor.")

        H = hessian_wrt_params(loss, params)
        H_np = H.numpy()
        p = H_np.shape[0]

        score_mat = score_obs_wrt_params_fast(model, loss_obs_builder)   # (n_obs, p)
        S_np = (score_mat.T @ score_mat).numpy()

        try:
            Hinv = np.linalg.inv(H_np + jitter * np.eye(p))
        except np.linalg.LinAlgError:
            if use_pinv:
                Hinv = np.linalg.pinv(H_np + jitter * np.eye(p))
            else:
                Hinv = np.linalg.inv(H_np + (10 * jitter) * np.eye(p))

        cov_np = Hinv @ S_np @ Hinv

        theta_hat = _flatten_params(params).detach().cpu().numpy()
        se = np.sqrt(np.clip(np.diag(cov_np), a_min=0.0, a_max=None))
        z, pvals = _safe_z_and_p(theta_hat, se)

    model._theta_hat = theta_hat
    model._hessian = H_np
    model._score_outer = S_np
    model._cov = cov_np
    model._se = se
    model._z = z
    model._p = pvals
    model._param_slices = _param_slices(model)
    model._cov_type = "sandwich"
    return model


def hessian_wrt_params(loss, params):
    params = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    g_flat = torch.cat([g.reshape(-1) for g in grads])
    P = g_flat.numel()

    H_rows = []
    for i in range(P):
        gi = g_flat[i]
        row_grads = torch.autograd.grad(gi, params, retain_graph=True, allow_unused=True)
        row = []
        for rg, p in zip(row_grads, params):
            if rg is None:
                row.append(torch.zeros_like(p).reshape(-1))
            else:
                row.append(rg.reshape(-1))
        H_rows.append(torch.cat(row))

    H = torch.stack(H_rows, dim=0)
    return H.detach().cpu().double()

def compute_hessian_cov(
    model: torch.nn.Module,
    loss_fn,
    *,
    device: str = "cpu",
    jitter: float = 1e-6,
    use_pinv: bool = True,
):
    model.to(device)
    with torch.enable_grad():
        params = _get_trainable_params(model)
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found.")

        loss = loss_fn()
        if loss.dim() != 0:
            raise ValueError("loss_fn must return a scalar tensor.")

        H = hessian_wrt_params(loss, params)
        H_np = H.numpy()
        p = H_np.shape[0]

        theta_hat = _flatten_params(params).detach().cpu().numpy()

        try:
            cov_np = np.linalg.inv(H_np + jitter * np.eye(p))
        except np.linalg.LinAlgError:
            if use_pinv:
                cov_np = np.linalg.pinv(H_np + jitter * np.eye(p))
            else:
                cov_np = np.linalg.inv(H_np + (10 * jitter) * np.eye(p))

        se = np.sqrt(np.clip(np.diag(cov_np), a_min=0.0, a_max=None))
        z, pvals = _safe_z_and_p(theta_hat, se)

    model._theta_hat = theta_hat
    model._hessian = H_np
    model._cov = cov_np
    model._se = se
    model._z = z
    model._p = pvals
    model._param_slices = _param_slices(model)
    return model


# ----------------------------
# Summary
# ----------------------------

def attach_summary(model: torch.nn.Module, *, coef_names: list[str], model_kind: str):
    def summary(self):
        import pandas as pd

        if not hasattr(self, "_cov"):
            raise RuntimeError("No covariance found. Run compute_hessian_cov first.")

        rows = []
        th = self._theta_hat
        se = self._se
        z = self._z
        p = self._p

        def _xnames(k: int):
            if len(coef_names) != k:
                raise ValueError(f"coef_names length {len(coef_names)} != beta/gamma length {k}")
            return coef_names

        for pname, sl, shape in self._param_slices:
            v = th[sl].reshape(shape)
            s = se[sl].reshape(shape)
            zv = z[sl].reshape(shape)
            pv = p[sl].reshape(shape)

            if pname == "beta":
                names = _xnames(v.size)
                flatv, flats, flatz, flatp = v.reshape(-1), s.reshape(-1), zv.reshape(-1), pv.reshape(-1)
                for i, nm in enumerate(names):
                    rows.append((f"count:{nm}", float(flatv[i]), float(flats[i]), float(flatz[i]), float(flatp[i])))

            elif (model_kind in ("zinb", "mzinb")) and (pname == "gamma"):
                names = _xnames(v.size)
                flatv, flats, flatz, flatp = v.reshape(-1), s.reshape(-1), zv.reshape(-1), pv.reshape(-1)
                for i, nm in enumerate(names):
                    rows.append((f"zero:{nm}", float(flatv[i]), float(flats[i]), float(flatz[i]), float(flatp[i])))

            elif pname == "log_theta":
                rows.append(("log(theta)", float(v), float(s), float(zv), float(pv)))

            else:
                flatv, flats, flatz, flatp = v.reshape(-1), s.reshape(-1), zv.reshape(-1), pv.reshape(-1)
                for i in range(flatv.size):
                    rows.append((f"{pname}[{i}]", float(flatv[i]), float(flats[i]), float(flatz[i]), float(flatp[i])))

        df = pd.DataFrame(rows, columns=["term", "Estimate", "Std. Error", "z value", "Pr(>|z|)"]).set_index("term")
        return df

    model.summary = types.MethodType(summary, model)
    return model


# ----------------------------
# LRT
# ----------------------------

def as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def lrt_test(ll_full: float, ll_reduced: float, df: int):
    LR = max(0.0, 2.0 * (ll_full - ll_reduced))
    chi2 = torch.distributions.chi2.Chi2(df=torch.as_tensor(float(df), dtype=torch.float64))
    p = 1.0 - float(chi2.cdf(torch.as_tensor(LR, dtype=torch.float64)).item())
    return float(LR), float(p)

def compute_ll_from_loss(loss_scalar: float) -> float:
    return -float(loss_scalar)

def attach_lrt(model_obj, *, lrt_results: dict):
    """
    lrt_results: dict[term] -> {
        "LR": float, "df": int, "p_value": float,
        "ll_full": float, "ll_reduced": float,
        "dropped": list[str] (optional)
    }
    attaches:
      model_obj._lrt_df : pandas.DataFrame (index=term)
      model_obj.lrt(term=None|str|list[str]) -> DataFrame or Series
    """
    # dict -> tidy dataframe
    rows = []
    for term, d in lrt_results.items():
        rows.append({
            "term": term,
            "LR": float(d.get("LR", float("nan"))),
            "df": int(d.get("df", 0)),
            "p_value": float(d.get("p_value", float("nan"))),
            "ll_full": float(d.get("ll_full", float("nan"))),
            "ll_reduced": float(d.get("ll_reduced", float("nan")))
        })

    lrt_df = pd.DataFrame(rows).set_index("term").sort_index()
    model_obj._lrt_df = lrt_df

    def lrt(self, term=None):
        """
        term:
          - None -> full DataFrame
          - str  -> row (Series)
          - list/tuple -> subset DataFrame
        """
        if term is None:
            return self._lrt_df

        if isinstance(term, str):
            if term not in self._lrt_df.index:
                raise KeyError(f"{term} not in LRT results. Available: {list(self._lrt_df.index)}")
            return self._lrt_df.loc[term]

        if isinstance(term, (list, tuple)):
            missing = [t for t in term if t not in self._lrt_df.index]
            if missing:
                raise KeyError(f"Missing terms in LRT results: {missing}. Available: {list(self._lrt_df.index)}")
            return self._lrt_df.loc[list(term)]

        raise TypeError("term must be None, str, or list/tuple[str]")

    model_obj.lrt = types.MethodType(lrt, model_obj)
    return model_obj