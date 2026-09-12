from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
ETA_MIN = -20.0
ETA_MAX = 20.0
LOG_THETA_MIN = -10.0
LOG_THETA_MAX = 10.0

def nb_log_prob(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Negative Binomial log PMF with mean mu and inverse-dispersion theta (>0).
    Parameterization used by many scRNA tools:
      var = mu + mu^2 / theta
    y, mu: (n,) ; theta: scalar or (n,)
    """
    y = y.to(mu.dtype)
    theta = torch.clamp(theta, min=EPS)
    mu = torch.clamp(mu, min=EPS)

    # log Gamma(y+theta) - log Gamma(theta) - log Gamma(y+1)
    lg = torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1.0)
    # theta*log(theta/(theta+mu)) + y*log(mu/(theta+mu))
    log_p = theta * (torch.log(theta) - torch.log(theta + mu)) + y * (torch.log(mu) - torch.log(theta + mu))
    return lg + log_p

def zinb_log_prob(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """
    Zero-Inflated NB:
      P(Y=0) = pi + (1-pi)*NB(0)
      P(Y>0) = (1-pi)*NB(y)
    pi in (0,1)
    """
    pi = torch.clamp(pi, min=EPS, max=1.0 - EPS)
    nb_lp = nb_log_prob(y, mu, theta)

    is_zero = (y == 0)
    # log( pi + (1-pi)*exp(nb_lp0) ) for zeros
    nb_lp0 = nb_log_prob(torch.zeros_like(y), mu, theta)
    zero_lp = torch.log(pi + (1.0 - pi) * torch.exp(nb_lp0) + EPS)
    nonzero_lp = torch.log(1.0 - pi + EPS) + nb_lp
    return torch.where(is_zero, zero_lp, nonzero_lp)

def mzinb_log_prob(y: torch.Tensor, nu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """
    Marginalized Zero-Inflated NB:
      P(Y=0) = pi + (1-pi)*NB(0)
      P(Y>0) = (1-pi)*NB(y)
    pi in (0,1)
    Here, nu = (1-pi)*mu is modeled directly.
    """
    pi = torch.clamp(pi, min=EPS, max=1.0 - EPS)
    nu = torch.clamp(nu, min=EPS)
    mu = nu / torch.clamp(1 - pi, min=EPS)

    nb_lp = nb_log_prob(y, mu, theta)

    is_zero = (y == 0)
    nb_lp0 = nb_log_prob(torch.zeros_like(y), mu, theta)
    zero_lp = torch.log(pi + (1.0 - pi) * torch.exp(nb_lp0) + EPS)
    nonzero_lp = torch.log(1.0 - pi + EPS) + nb_lp
    return torch.where(is_zero, zero_lp, nonzero_lp)



@dataclass
class GLMConfig:
    link: str = "log"           # mean link for mu
    zi_link: str = "logit"      # link for pi
    offset: bool = True         # use log library size as offset
    ridge: float = 0.0          # L2 penalty weight (optional)

class NBGLM(nn.Module):
    """
    Gene-wise NB GLM:
      log(mu_i) = offset_i + x_i^T beta
      theta > 0
    """
    def __init__(self, p: int, cfg: GLMConfig = GLMConfig()):
        super().__init__()
        self.cfg = cfg
        self.beta = nn.Parameter(torch.zeros(p))
        self.log_theta = nn.Parameter(torch.tensor(0.0))  # theta = exp(log_theta)

    def forward(self, y, X, log_offset=None):
        return self.neg_log_lik_obs(y, X, log_offset=log_offset)
    
    def forward_mu(self, X: torch.Tensor, log_offset: torch.Tensor | None) -> torch.Tensor:
        eta = X @ self.beta
        if self.cfg.offset and log_offset is not None:
            eta = eta + log_offset
        eta = torch.clamp(eta, min=ETA_MIN, max=ETA_MAX)
        return torch.exp(eta)

    def neg_log_lik_obs(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        log_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        mu = self.forward_mu(X, log_offset)
        theta = torch.exp(torch.clamp(self.log_theta, min=LOG_THETA_MIN, max=LOG_THETA_MAX))
        ll_i = nb_log_prob(y, mu, theta)   # shape (n,)

        if self.cfg.ridge > 0:
            # penalty는 observation-wise가 아니라 total loss에만 넣는 게 일반적
            return -ll_i
        return -ll_i

    def neg_log_lik(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        log_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        nll_i = self.neg_log_lik_obs(y, X, log_offset=log_offset)
        loss = nll_i.sum()

        if self.cfg.ridge > 0:
            loss = loss + self.cfg.ridge * (self.beta**2).sum()
        return loss

class ZINBGLM(nn.Module):
    """
    Gene-wise ZINB GLM:
      log(mu_i) = offset_i + x_i^T beta
      logit(pi_i) = z_i^T gamma   (often z == X)
      theta > 0
    """
    def __init__(self, p_mean: int, p_zi: int | None = None, cfg: GLMConfig = GLMConfig()):
        super().__init__()
        self.cfg = cfg
        self.p_mean = p_mean
        self.p_zi = p_zi if p_zi is not None else p_mean

        self.beta = nn.Parameter(torch.zeros(self.p_mean))
        self.gamma = nn.Parameter(torch.zeros(self.p_zi))
        self.log_theta = nn.Parameter(torch.tensor(0.0))

    def forward(self, y, X, Z=None, log_offset=None):
        return self.neg_log_lik_obs(y, X, Z=Z, log_offset=log_offset)
        
    def forward_mu(self, X: torch.Tensor, log_offset: torch.Tensor | None) -> torch.Tensor:
        eta = X @ self.beta
        if self.cfg.offset and log_offset is not None:
            eta = eta + log_offset
        eta = torch.clamp(eta, min=ETA_MIN, max=ETA_MAX)
        return torch.exp(eta)

    def forward_pi(self, Z: torch.Tensor) -> torch.Tensor:
        zi_eta = Z @ self.gamma
        zi_eta = torch.clamp(zi_eta, min=-20.0, max=20.0)
        return torch.sigmoid(zi_eta)

    def neg_log_lik_obs(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        Z: torch.Tensor | None = None,
        log_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        if Z is None:
            Z = X
        mu = self.forward_mu(X, log_offset)
        pi = self.forward_pi(Z)
        theta = torch.exp(torch.clamp(self.log_theta, min=LOG_THETA_MIN, max=LOG_THETA_MAX))
        ll_i = zinb_log_prob(y, mu, theta, pi)   # shape (n,)
        return -ll_i

    def neg_log_lik(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        Z: torch.Tensor | None = None,
        log_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        nll_i = self.neg_log_lik_obs(y, X, Z=Z, log_offset=log_offset)
        loss = nll_i.sum()

        if self.cfg.ridge > 0:
            loss = loss + self.cfg.ridge * ((self.beta**2).sum() + (self.gamma**2).sum())
        return loss
    
class MZINBGLM(nn.Module):
    """
    Gene-wise Marginalized ZINB GLM:
      log(nu_i) = offset_i + x_i^T beta
      logit(pi_i) = z_i^T gamma
      theta > 0
    """
    def __init__(self, p_mean: int, p_zi: int | None = None, cfg: GLMConfig = GLMConfig()):
        super().__init__()
        self.cfg = cfg
        self.p_mean = p_mean
        self.p_zi = p_zi if p_zi is not None else p_mean

        self.beta = nn.Parameter(torch.zeros(self.p_mean))
        self.gamma = nn.Parameter(torch.zeros(self.p_zi))
        self.log_theta = nn.Parameter(torch.tensor(0.0))

    def forward(self, y, X, Z=None, log_offset=None):
        return self.neg_log_lik_obs(y, X, Z=Z, log_offset=log_offset)
        
    def forward_nu(self, X: torch.Tensor, log_offset: torch.Tensor | None) -> torch.Tensor:
        eta = X @ self.beta
        if self.cfg.offset and log_offset is not None:
            eta = eta + log_offset
        eta = torch.clamp(eta, min=ETA_MIN, max=ETA_MAX)
        return torch.exp(eta)

    def forward_pi(self, Z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(Z @ self.gamma)

    def neg_log_lik_obs(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        Z: torch.Tensor | None = None,
        log_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        if Z is None:
            Z = X
        nu = self.forward_nu(X, log_offset)
        pi = self.forward_pi(Z)
        theta = torch.exp(torch.clamp(self.log_theta, min=LOG_THETA_MIN, max=LOG_THETA_MAX))
        ll_i = mzinb_log_prob(y, nu, theta, pi)   # shape (n,)
        return -ll_i

    def neg_log_lik(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        Z: torch.Tensor | None = None,
        log_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        nll_i = self.neg_log_lik_obs(y, X, Z=Z, log_offset=log_offset)
        loss = nll_i.sum()

        if self.cfg.ridge > 0:
            loss = loss + self.cfg.ridge * ((self.beta**2).sum() + (self.gamma**2).sum())
        return loss
    

class CZINBGLM(nn.Module):
    def __init__(self, n_genes: int, p_mean: int, p_zi: int | None = None, cfg: GLMConfig = GLMConfig()):
        super().__init__()
        self.cfg = cfg
        self.G = n_genes
        self.p_mean = p_mean
        self.p_zi = p_zi if p_zi is not None else p_mean

        self.beta = nn.Parameter(torch.zeros(self.G, self.p_mean))
        self.log_theta = nn.Parameter(torch.zeros(self.G))
        self.gamma = nn.Parameter(torch.zeros(self.p_zi))

    def forward_mu(self, X, log_offset):
        eta = X @ self.beta.T
        if self.cfg.offset and log_offset is not None:
            eta = eta + log_offset.view(-1, 1)
        eta = torch.clamp(eta, min=ETA_MIN, max=ETA_MAX)
        return torch.exp(eta)

    def forward_pi(self, Z):
        zi_eta = Z @ self.gamma
        zi_eta = torch.clamp(zi_eta, min=ETA_MIN, max=ETA_MAX)
        return torch.sigmoid(zi_eta).view(-1, 1)

    def neg_log_lik_obs(self, Y, X, Z=None, log_offset=None):
        if Z is None:
            Z = X
        mu = self.forward_mu(X, log_offset)              # (n,G)
        pi = self.forward_pi(Z).reshape(-1)             # (n,)
        theta = torch.exp(
            torch.clamp(self.log_theta, min=LOG_THETA_MIN, max=LOG_THETA_MAX)
        ).view(1, -1)                                   # (1,G)

        nb_ll_mat = nb_log_prob(Y, mu, theta)           # (n,G)
        nb_ll_sum = nb_ll_mat.sum(dim=1)                # (n,)
        all_zero = (Y == 0).all(dim=1)

        log_pi = torch.log(torch.clamp(pi, min=EPS))
        log_one_minus_pi = torch.log(torch.clamp(1.0 - pi, min=EPS))

        ll_all_zero = torch.logaddexp(log_pi, log_one_minus_pi + nb_ll_sum)
        ll_nonzero = log_one_minus_pi + nb_ll_sum
        ll = torch.where(all_zero, ll_all_zero, ll_nonzero)
        return -ll                                      # (n,)

    def neg_log_lik(self, Y, X, Z=None, log_offset=None):
        nll_i = self.neg_log_lik_obs(Y, X, Z=Z, log_offset=log_offset)
        loss = nll_i.sum()

        if self.cfg.ridge > 0:
            loss = loss + self.cfg.ridge * ((self.beta**2).sum() + (self.gamma**2).sum())
        return loss

    def forward(self, Y, X, Z=None, log_offset=None):
        return self.neg_log_lik_obs(Y, X, Z=Z, log_offset=log_offset)
