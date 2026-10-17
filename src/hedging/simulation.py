"""Monte Carlo simulation of a short call hedged in discrete time."""
from dataclasses import dataclass

import numpy as np

from .black_scholes import call_delta, call_gamma, call_price


@dataclass
class Market:
    """Contract and stock parameters (time in years)."""
    S0: float = 100.0
    K: float = 100.0
    T: float = 0.25
    r: float = 0.02
    mu: float = 0.08      # real-world drift
    sigma: float = 0.20   # true (realised) volatility


def simulate_gbm(market, N, M, rng):
    """M exact paths of geometric Brownian motion on N equal steps, shape (M, N+1)."""
    dt = market.T / N
    Z = rng.standard_normal((M, N))
    increments = (market.mu - 0.5 * market.sigma**2) * dt + market.sigma * np.sqrt(dt) * Z
    S = np.empty((M, N + 1))
    S[:, 0] = market.S0
    S[:, 1:] = market.S0 * np.exp(np.cumsum(increments, axis=1))
    return S


def hedge_pnl(market, N, M=20_000, k=0.0, sigma_hedge=None, sigma_price=None,
              seed=0, return_paths=False):
    """Terminal P&L of selling one call and delta-hedging it at N equally spaced dates.

    k            round-trip proportional cost: trading n shares at S costs (k/2)|n| S.
    sigma_hedge  volatility used to compute the hedge ratio (default: true sigma).
    sigma_price  volatility at which the option is sold (default: sigma_hedge).
    """
    sh = market.sigma if sigma_hedge is None else sigma_hedge
    sp = sh if sigma_price is None else sigma_price
    rng = np.random.default_rng(seed)
    S = simulate_gbm(market, N, M, rng)
    dt = market.T / N
    growth = np.exp(market.r * dt)

    delta = np.full(M, call_delta(market.S0, market.K, market.T, market.r, sh))
    cash = (call_price(market.S0, market.K, market.T, market.r, sp)
            - delta * market.S0 - 0.5 * k * np.abs(delta) * market.S0)
    for i in range(1, N):
        cash *= growth
        new_delta = call_delta(S[:, i], market.K, market.T - i * dt, market.r, sh)
        trade = new_delta - delta
        cash -= trade * S[:, i] + 0.5 * k * np.abs(trade) * S[:, i]
        delta = new_delta
    cash *= growth

    pnl = cash + delta * S[:, N] - np.maximum(S[:, N] - market.K, 0.0)
    return (pnl, S) if return_paths else pnl


def gamma_pnl_prediction(market, S, sigma_implied):
    """Leading-order P&L of hedging at the implied volatility, evaluated on each path:

        0.5 * sum_i exp(r (T - t_i)) Gamma_i S_i^2 (sigma_implied^2 - sigma^2) dt,

    with Gamma computed at the implied volatility.
    """
    M, N1 = S.shape
    N = N1 - 1
    dt = market.T / N
    t = np.arange(N) * dt
    gamma = call_gamma(S[:, :-1], market.K, market.T - t, market.r, sigma_implied)
    weights = np.exp(market.r * (market.T - t))
    return 0.5 * (sigma_implied**2 - market.sigma**2) * dt * np.sum(
        weights * gamma * S[:, :-1] ** 2, axis=1)


def leland_sigma(sigma, k, dt):
    """Leland (1985) volatility adjusted for round-trip proportional cost k."""
    return sigma * np.sqrt(1.0 + np.sqrt(2.0 / np.pi) * k / (sigma * np.sqrt(dt)))
