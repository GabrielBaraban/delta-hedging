"""Black-Scholes formulas for a European call (vectorised with NumPy)."""
import numpy as np
from scipy.stats import norm

_TAU_MIN = 1e-12


def _d1_d2(S, K, tau, r, sigma):
    tau = np.maximum(tau, _TAU_MIN)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / vol
    return d1, d1 - vol


def call_price(S, K, tau, r, sigma):
    """Black-Scholes price of a European call with time to maturity tau."""
    d1, d2 = _d1_d2(S, K, tau, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * np.maximum(tau, 0.0)) * norm.cdf(d2)


def call_delta(S, K, tau, r, sigma):
    """dC/dS = Phi(d1)."""
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return norm.cdf(d1)


def call_gamma(S, K, tau, r, sigma):
    """d2C/dS2 = phi(d1) / (S sigma sqrt(tau))."""
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(np.maximum(tau, _TAU_MIN)))


def call_vega(S, K, tau, r, sigma):
    """dC/dsigma = S phi(d1) sqrt(tau)."""
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(np.maximum(tau, 0.0))
