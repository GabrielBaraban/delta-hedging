"""Basic checks: Greeks against finite differences, and the hedging limits."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hedging import (Market, call_delta, call_gamma, call_price, call_vega,  # noqa: E402
                     gamma_pnl_prediction, hedge_pnl, leland_sigma)

S, K, TAU, R, SIG = 103.0, 100.0, 0.4, 0.02, 0.25


def test_delta_matches_finite_difference():
    h = 1e-4
    fd = (call_price(S + h, K, TAU, R, SIG) - call_price(S - h, K, TAU, R, SIG)) / (2 * h)
    assert call_delta(S, K, TAU, R, SIG) == pytest.approx(fd, rel=1e-6)


def test_gamma_matches_finite_difference():
    h = 1e-3
    fd = (call_delta(S + h, K, TAU, R, SIG) - call_delta(S - h, K, TAU, R, SIG)) / (2 * h)
    assert call_gamma(S, K, TAU, R, SIG) == pytest.approx(fd, rel=1e-5)


def test_vega_matches_finite_difference():
    h = 1e-5
    fd = (call_price(S, K, TAU, R, SIG + h) - call_price(S, K, TAU, R, SIG - h)) / (2 * h)
    assert call_vega(S, K, TAU, R, SIG) == pytest.approx(fd, rel=1e-6)


def test_price_at_maturity_is_payoff():
    assert call_price(110.0, K, 0.0, R, SIG) == pytest.approx(10.0, abs=1e-8)
    assert call_price(90.0, K, 0.0, R, SIG) == pytest.approx(0.0, abs=1e-8)


def test_frequent_hedging_replicates_the_call():
    pnl = hedge_pnl(Market(), N=2000, M=4000)
    assert abs(pnl.mean()) < 0.01
    assert pnl.std() < 0.1


def test_hedging_error_halves_when_n_quadruples():
    s1 = hedge_pnl(Market(), N=50, M=8000, seed=1).std()
    s2 = hedge_pnl(Market(), N=200, M=8000, seed=1).std()
    assert s1 / s2 == pytest.approx(2.0, rel=0.1)


def test_leland_volatility_exceeds_sigma_and_vanishes_without_costs():
    assert leland_sigma(0.2, 0.0, 0.01) == pytest.approx(0.2)
    assert leland_sigma(0.2, 0.004, 0.01) > 0.2


def test_gamma_formula_is_zero_when_vol_is_right():
    m = Market()
    _, paths = hedge_pnl(m, N=50, M=100, return_paths=True)
    assert np.allclose(gamma_pnl_prediction(m, paths, m.sigma), 0.0)
