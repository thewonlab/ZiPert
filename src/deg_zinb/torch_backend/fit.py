from __future__ import annotations
from dataclasses import dataclass
import math
import os
import random
from typing import Optional
import torch
import numpy as np

@dataclass
class FitConfig:
    method: str = "lbfgs"       # "adam" | "lbfgs" | "hybrid"
    lr: float = 1e-2

    max_iter: int = 2000
    rtol: float = 1e-6
    atol: float = 0.0

    lbfgs_lr: Optional[float] = None
    lbfgs_max_eval: Optional[int] = None
    lbfgs_tol_grad: Optional[float] = None
    lbfgs_tol_change: Optional[float] = None

    hybrid_adam_iters: Optional[int] = None
    hybrid_adam_lr: Optional[float] = None
    
    warm_start: bool = True
    warm_ridge: float = 1e-4
    warm_eps_y: float = 1e-1
    warm_noise_scale: float = 1e-4
    verbose: bool = False
    
    n_runs: int = 1

    # stability
    grad_clip_norm: Optional[float] = 10.0
    beta_init_clip: float = 5.0
    gamma_init_clip: float = 5.0
    log_theta_init_clip: float = 5.0
    check_finite_inputs: bool = True

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def _check_tensor_finite(name: str, x: Optional[torch.Tensor]):
    if x is None:
        return
    if not torch.isfinite(x).all():
        bad_n = int((~torch.isfinite(x)).sum().item())
        raise RuntimeError(f"Non-finite values found in {name}: n_bad={bad_n}")

def _check_inputs(y: Optional[torch.Tensor], X: Optional[torch.Tensor], log_offset: Optional[torch.Tensor]):
    _check_tensor_finite("y", y)
    _check_tensor_finite("X", X)
    _check_tensor_finite("log_offset", log_offset)

    if y is not None:
        if (y < 0).any():
            raise RuntimeError("Negative counts found in y.")
    if X is not None and X.ndim != 2:
        raise RuntimeError(f"X must be 2D, got shape={tuple(X.shape)}")
    if y is not None and X is not None and y.shape[0] != X.shape[0]:
        raise RuntimeError(f"Shape mismatch: len(y)={y.shape[0]} vs X.shape[0]={X.shape[0]}")
    if log_offset is not None and y is not None and log_offset.shape[0] != y.shape[0]:
        raise RuntimeError(f"Shape mismatch: len(log_offset)={log_offset.shape[0]} vs len(y)={y.shape[0]}")
    
def _converged(prev: float, cur: float, rtol: float, atol: float) -> bool:
    return abs(prev - cur) <= (atol + rtol * max(abs(prev), abs(cur), 1.0))

def _run_adam(model, loss_fn, lr: float, max_iter: int, verbose: bool,
              rtol: float | None = None, atol: float = 0.0,
              grad_clip_norm: Optional[float] = None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    prev = None

    for it in range(max_iter):
        opt.zero_grad()
        loss = loss_fn()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss before backward at iter={it}: {float(loss.detach().cpu())}")

        loss.backward()

        bad_grad_names = []
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad_names.append(name)
        if bad_grad_names:
            raise RuntimeError(f"Non-finite gradients at iter={it}: {bad_grad_names}")

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        opt.step()

        cur = float(loss.detach().cpu())
        if verbose and (it % 200 == 0):
            tag = "adam(warmup)" if rtol is None else "adam"
            print(f"[{tag}] iter={it} loss={cur:.6f}")

        if rtol is not None and prev is not None and _converged(prev, cur, rtol, atol):
            break
        prev = cur

    return model


def _run_lbfgs_one_shot(model, loss_fn, lr: float,
                        max_iter: int, max_eval: Optional[int],
                        tol_grad: float, tol_change: float,
                        verbose: bool):
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tol_grad,
        tolerance_change=tol_change,
        line_search_fn="strong_wolfe",
    )

    n_calls = 0

    def closure():
        nonlocal n_calls
        n_calls += 1
        opt.zero_grad()
        loss = loss_fn()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in LBFGS closure at eval={n_calls}: {float(loss.detach().cpu())}")

        loss.backward()

        bad_grad_names = []
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad_names.append(name)
        if bad_grad_names:
            raise RuntimeError(f"Non-finite gradients in LBFGS closure at eval={n_calls}: {bad_grad_names}")

        if verbose and (n_calls % 25 == 0):
            print(f"[lbfgs] eval={n_calls} loss={float(loss.detach().cpu()):.6f}")
        return loss

    final_loss = opt.step(closure)

    with torch.no_grad():
        true_final = float(loss_fn().detach().cpu())

    if verbose:
        print(f"[lbfgs] done eval={n_calls} step_return={float(final_loss.detach().cpu()):.6f} final_loss={true_final:.6f}")
    return model

def _logit(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


@torch.no_grad()
def _warm_start_params(model, y: torch.Tensor, X: torch.Tensor, log_offset: Optional[torch.Tensor] = None,
                       ridge: float = 1e-4, eps_y: float = 1e-1, noise_scale=0, seed=None,
                       beta_init_clip: float = 5.0,
                       gamma_init_clip: float = 5.0,
                       log_theta_init_clip: float = 5.0):
    """
    - beta: Poisson-like init via OLS on log(y+eps) - offset
    - log_theta: 0
    - gamma (if exists): intercept = logit(zero_rate), others 0
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)

    def _randn_like_with_gen(x, g=None):
        if g is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
    
    y = y.to(device=device, dtype=dtype)
    X = X.to(device=device, dtype=dtype)
    if log_offset is not None:
        log_offset = log_offset.to(device=device, dtype=dtype).reshape(-1)

    _check_inputs(y, X, log_offset)

    if hasattr(model, "log_theta") and model.log_theta is not None:
        model.log_theta.fill_(0.0)
        model.log_theta.clamp_(min=-log_theta_init_clip, max=log_theta_init_clip)

    if hasattr(model, "beta") and model.beta is not None:
        eta0 = torch.log(torch.clamp(y, min=0) + eps_y)
        if log_offset is not None:
            eta0 = eta0 - log_offset.view(-1, *([1] * (eta0.ndim - 1)))

        eta0 = torch.clamp(eta0, min=-20.0, max=20.0)

        p = X.shape[1]
        XtX = X.T @ X
        XtX = XtX + ridge * torch.eye(p, device=device, dtype=dtype)
        Xty = X.T @ eta0

        try:
            beta_init = torch.linalg.solve(XtX, Xty)
        except RuntimeError:
            beta_init = torch.linalg.lstsq(XtX, Xty).solution

        if model.beta.ndim == 1:
            beta_init = beta_init.reshape(-1)
        else:
            beta_init = beta_init.T

        beta_init = torch.clamp(beta_init, min=-beta_init_clip, max=beta_init_clip)

        if noise_scale > 0:
            beta_init = beta_init + noise_scale * _randn_like_with_gen(beta_init, g)

        model.beta.copy_(beta_init)

    if hasattr(model, "gamma") and model.gamma is not None:
        model.gamma.zero_()
        if y.ndim == 1:
            zero_rate = (y == 0).to(dtype=dtype).mean()
        else:
            zero_rate = (y == 0).all(dim=1).to(dtype=dtype).mean()
        gamma0 = _logit(zero_rate).reshape(())
        gamma0 = torch.clamp(gamma0, min=-gamma_init_clip, max=gamma_init_clip)
        model.gamma.view(-1)[0] = gamma0

        if noise_scale > 0:
            model.gamma += noise_scale * _randn_like_with_gen(model.gamma, g)

        model.gamma.clamp_(min=-gamma_init_clip, max=gamma_init_clip)
        
def fit_model(model, loss_fn, cfg: FitConfig, device: str = "cpu",
              *, y: Optional[torch.Tensor] = None,
              X: Optional[torch.Tensor] = None,
              log_offset: Optional[torch.Tensor] = None,
              seed: Optional[int] = None, 
              deterministic: bool = False,
              dtype=torch.float64):
    if seed is not None:
        set_seed(seed, deterministic=deterministic)

    model.to(device=device, dtype=dtype)
    y_ = None if y is None else y.to(device=device, dtype=dtype)
    X_ = None if X is None else X.to(device=device, dtype=dtype)
    log_offset_ = None if log_offset is None else log_offset.to(device=device, dtype=dtype)

    if getattr(cfg, "check_finite_inputs", True):
        _check_inputs(y_, X_, log_offset_)

    if cfg.warm_start:
        if (y is not None) and (X is not None):
            _warm_start_params(
                model,
                y=y_,
                X=X_,
                log_offset=log_offset_,
                ridge=cfg.warm_ridge,
                eps_y=cfg.warm_eps_y,
                noise_scale=cfg.warm_noise_scale,
                seed=seed,
                beta_init_clip=cfg.beta_init_clip,
                gamma_init_clip=cfg.gamma_init_clip,
                log_theta_init_clip=cfg.log_theta_init_clip,
            )

    with torch.no_grad():
        init_loss = loss_fn()
        if not torch.isfinite(init_loss):
            raise RuntimeError(f"Initial loss is non-finite before optimization: {float(init_loss.detach().cpu())}")

    m = cfg.method.lower()

    lbfgs_lr = cfg.lbfgs_lr if cfg.lbfgs_lr is not None else cfg.lr
    lbfgs_tol_grad = cfg.lbfgs_tol_grad if cfg.lbfgs_tol_grad is not None else cfg.rtol
    lbfgs_tol_change = cfg.lbfgs_tol_change if cfg.lbfgs_tol_change is not None else (cfg.atol + cfg.rtol)

    if m == "adam":
        return _run_adam(
            model, loss_fn,
            lr=cfg.lr,
            max_iter=cfg.max_iter,
            verbose=cfg.verbose,
            rtol=cfg.rtol,
            atol=cfg.atol,
            grad_clip_norm=cfg.grad_clip_norm,
        )

    if m == "lbfgs":
        return _run_lbfgs_one_shot(
            model, loss_fn,
            lr=lbfgs_lr,
            max_iter=cfg.max_iter,
            max_eval=cfg.lbfgs_max_eval,
            tol_grad=lbfgs_tol_grad,
            tol_change=lbfgs_tol_change,
            verbose=cfg.verbose,
        )

    if m == "hybrid":
        warm_iters = cfg.hybrid_adam_iters if cfg.hybrid_adam_iters is not None else min(300, cfg.max_iter // 4)
        warm_lr = cfg.hybrid_adam_lr if cfg.hybrid_adam_lr is not None else cfg.lr

        _run_adam(
            model, loss_fn,
            lr=warm_lr,
            max_iter=warm_iters,
            verbose=cfg.verbose,
            rtol=None,
            atol=cfg.atol,
            grad_clip_norm=cfg.grad_clip_norm,
        )

        with torch.no_grad():
            mid_loss = loss_fn()
            if not torch.isfinite(mid_loss):
                raise RuntimeError(f"Loss became non-finite after Adam warmup: {float(mid_loss.detach().cpu())}")

        return _run_lbfgs_one_shot(
            model, loss_fn,
            lr=lbfgs_lr,
            max_iter=max(1, cfg.max_iter - warm_iters),
            max_eval=cfg.lbfgs_max_eval,
            tol_grad=lbfgs_tol_grad,
            tol_change=lbfgs_tol_change,
            verbose=cfg.verbose,
        )

    raise ValueError(f"Unknown optimization method: {cfg.method}")
