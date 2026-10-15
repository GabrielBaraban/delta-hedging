"""Runs every experiment of the paper.

Writes the figures to figures/ (PDF for the paper, PNG for the README) and the
numbers quoted in the text to paper/numbers.tex, so the paper never contains a
hand-typed result.

Usage (from the repository root):  python experiments/run_experiments.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hedging import (Market, call_delta, call_price, call_vega,  # noqa: E402
                     gamma_pnl_prediction, hedge_pnl, leland_sigma)

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "lines.markersize": 5, "font.size": 9.5,
    "legend.frameon": False, "mathtext.fontset": "cm",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=180)
    plt.close(fig)


def plain_log_axis(axis):
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


numbers = {}  # LaTeX macro name -> formatted value


def num(name, value, fmt="{:.2f}"):
    numbers[name] = fmt.format(value)


mkt = Market()
C0 = call_price(mkt.S0, mkt.K, mkt.T, mkt.r, mkt.sigma)
D0 = call_delta(mkt.S0, mkt.K, mkt.T, mkt.r, mkt.sigma)
vega0 = call_vega(mkt.S0, mkt.K, mkt.T, mkt.r, mkt.sigma)
num("CallPrice", C0); num("InitDelta", D0, "{:.3f}")
M_MAIN, M_SWEEP = 20_000, 10_000
num("Mmain", M_MAIN, "{:,}"); num("Msweep", M_SWEEP, "{:,}")
numbers["Mmain"] = numbers["Mmain"].replace(",", "{,}")
numbers["Msweep"] = numbers["Msweep"].replace(",", "{,}")

# ------------------------------------------------- 1. discrete rebalancing
print("1. discrete rebalancing")
hist_N = [(12, ORANGE, "weekly"), (63, AQUA, "daily"), (252, BLUE, "4 times a day")]
fig, ax = plt.subplots(figsize=(6.2, 3.1))
bins = np.linspace(-3, 3, 121)
for N, c, lab in hist_N:
    p = hedge_pnl(mkt, N, M=M_MAIN)
    ax.hist(p, bins=bins, histtype="step", lw=1.6, color=c, density=True,
            label=f"$N={N}$ ({lab}), std {p.std():.2f}")
    tag = {12: "Weekly", 63: "Daily", 252: "Intraday"}[N]
    num(f"Std{tag}", p.std()); num(f"Mean{tag}", p.mean(), "{:+.3f}")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("Terminal P&L of the hedger"); ax.set_ylabel("Density")
ax.legend(loc="upper right")
save(fig, "fig1_pnl_hist")

Ns = np.array([4, 8, 16, 32, 64, 128, 256, 512])
stds = np.array([hedge_pnl(mkt, N, M=M_SWEEP, seed=1).std() for N in Ns])
dk = np.sqrt(np.pi / 4) * mkt.sigma * vega0 / np.sqrt(Ns)
slope = np.polyfit(np.log(Ns), np.log(stds), 1)[0]
num("Slope", slope, "{:.3f}")
num("DKmaxRelErr", 100 * np.max(np.abs(stds / dk - 1)), "{:.0f}")
fig, ax = plt.subplots(figsize=(5.2, 3.1))
ax.loglog(Ns, stds, "o-", color=BLUE, label="Simulation")
ax.loglog(Ns, dk, "--", color=ORANGE, label="Derman–Kamal approximation")
plain_log_axis(ax.xaxis); plain_log_axis(ax.yaxis)
ax.set_yticks([0.2, 0.3, 0.5, 1.0, 1.5])
ax.set_xlabel("Number of rebalancing dates $N$"); ax.set_ylabel("Std of terminal P&L")
ax.legend()
save(fig, "fig2_convergence")

drift_bias = {}
for m_ in (-0.20, 0.40):
    drift_bias[m_] = [hedge_pnl(Market(mu=m_), N, M=M_MAIN).mean() for N in (63, 1000)]
num("DriftBiasDaily", max(abs(v[0]) for v in drift_bias.values()), "{:.3f}")
num("DriftBiasFine", max(abs(v[1]) for v in drift_bias.values()), "{:.4f}")

# ------------------------------------------------------ 2. transaction costs
print("2. transaction costs")
k = 0.004
num("kpct", 100 * k, "{:.1f}")
Ns_tc = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
mean_tc, std_tc = [], []
for N in Ns_tc:
    p = hedge_pnl(mkt, N, M=M_SWEEP, k=k, seed=2)
    mean_tc.append(p.mean()); std_tc.append(p.std())
mean_tc, std_tc = np.array(mean_tc), np.array(std_tc)
rms_tc = np.sqrt(mean_tc**2 + std_tc**2)
best = int(Ns_tc[np.argmin(rms_tc)])
num("BestN", best, "{:d}"); num("BestRMS", rms_tc.min())
num("CostLossMaxN", -mean_tc[-1]); num("MaxNtc", Ns_tc[-1], "{:d}")
num("StdMinTc", std_tc.min()); num("StdMinTcN", Ns_tc[np.argmin(std_tc)], "{:d}")
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.semilogx(Ns_tc, -mean_tc, "o-", color=ORANGE, label="Expected cost: $-$mean P&L")
ax.semilogx(Ns_tc, std_tc, "o-", color=BLUE, label="Risk: std of P&L")
ax.semilogx(Ns_tc, rms_tc, "o-", color=AQUA, label="Root-mean-square P&L")
ax.set_ylim(0, 3.0)
ax.plot([best, best], [0, rms_tc.min()], color=INK2, lw=0.8, ls=":")
ax.text(best * 1.08, 0.2, f"minimum at $N={best}$", color=INK2)
plain_log_axis(ax.xaxis)
ax.set_xlabel("Number of rebalancing dates $N$"); ax.set_ylabel("P&L units")
ax.legend(loc="upper left")
save(fig, "fig3_costs")

Ns_l = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
bs_m, bs_s, le_m, le_s = [], [], [], []
for N in Ns_l:
    sl = leland_sigma(mkt.sigma, k, mkt.T / N)
    a = hedge_pnl(mkt, N, M=M_SWEEP, k=k, seed=3)
    b = hedge_pnl(mkt, N, M=M_SWEEP, k=k, sigma_hedge=sl, seed=3)
    bs_m.append(a.mean()); bs_s.append(a.std()); le_m.append(b.mean()); le_s.append(b.std())
setup = 0.5 * k * D0 * mkt.S0
num("SetupCost", setup, "{:.3f}")
num("LelandMeanMin", min(le_m), "{:+.3f}"); num("LelandMeanMax", max(le_m), "{:+.3f}")
num("LelandStdMaxN", le_s[-1]); num("BSStdMaxN", bs_s[-1])
sl_daily = leland_sigma(mkt.sigma, k, mkt.T / 64)
num("LelandSigDaily", 100 * sl_daily, "{:.1f}")
num("LelandPriceDaily", call_price(mkt.S0, mkt.K, mkt.T, mkt.r, sl_daily))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, a, b, title in [(axes[0], bs_m, le_m, "Mean of terminal P&L"),
                        (axes[1], bs_s, le_s, "Std of terminal P&L")]:
    ax.semilogx(Ns_l, a, "o-", color=ORANGE, label="Hedge at $\\sigma$ (Black–Scholes)")
    ax.semilogx(Ns_l, b, "o-", color=BLUE, label="Hedge at $\\hat\\sigma$ (Leland)")
    plain_log_axis(ax.xaxis)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xlabel("Number of rebalancing dates $N$")
axes[0].axhline(0, color=INK2, lw=0.8)
axes[0].legend(loc="lower left")
save(fig, "fig4_leland")

# ---------------------------------------------------- 3. wrong volatility
print("3. wrong volatility")
sig_imp, sig_real = 0.20, 0.15
num("SigImp", 100 * sig_imp, "{:.0f}"); num("SigReal", 100 * sig_real, "{:.0f}")
mw = Market(sigma=sig_real)
const = np.exp(mw.r * mw.T) * (call_price(mw.S0, mw.K, mw.T, mw.r, sig_imp)
                               - call_price(mw.S0, mw.K, mw.T, mw.r, sig_real))
num("PriceGap", const, "{:.3f}")

N_fine = 1000
p_imp, S = hedge_pnl(mw, N_fine, M=M_MAIN, sigma_hedge=sig_imp, return_paths=True, seed=4)
pred = gamma_pnl_prediction(mw, S, sig_imp)
p_real = hedge_pnl(mw, N_fine, M=M_MAIN, sigma_hedge=sig_real, sigma_price=sig_imp, seed=4)
num("Nfine", N_fine, "{:d}")
num("CorrFine", np.corrcoef(p_imp, pred)[0, 1], "{:.3f}")
num("ResidStdFine", (p_imp - pred).std(), "{:.3f}")
num("ImpMean", p_imp.mean(), "{:.3f}"); num("ImpStd", p_imp.std(), "{:.3f}")
num("RealMean", p_real.mean(), "{:.3f}"); num("RealStd", p_real.std(), "{:.3f}")
num("ImpLossProb", 100 * np.mean(p_imp < 0), "{:.1f}")
num("ImpMin", p_imp.min(), "{:.2f}"); num("ImpMax", p_imp.max(), "{:.2f}")

rng = np.random.default_rng(0)
idx = rng.choice(M_MAIN, 4000, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
ax = axes[0]
ax.scatter(pred[idx], p_imp[idx], s=4, color=BLUE, alpha=0.35, linewidths=0)
lim = [min(pred.min(), p_imp.min()), max(pred.max(), p_imp.max())]
ax.plot(lim, lim, color=ORANGE, lw=1.2, label="45° line")
ax.set_xlabel("Predicted by the gamma formula")
ax.set_ylabel("Simulated P&L")
ax.set_title(f"Hedge at implied vol, $N={N_fine}$", loc="left", fontsize=9.5)
ax.legend(loc="upper left")
ax = axes[1]
bins = np.linspace(-0.5, 2.5, 121)
ax.hist(p_imp, bins=bins, histtype="step", lw=1.6, color=BLUE, density=True,
        label=f"Hedge at implied vol (std {p_imp.std():.2f})")
ax.hist(p_real, bins=bins, histtype="step", lw=1.6, color=ORANGE, density=True,
        label=f"Hedge at realised vol (std {p_real.std():.2f})")
ax.plot([const, const], [0, 6.6], color=INK2, lw=0.8, ls=":")
ax.set_xlabel("Terminal P&L"); ax.set_ylabel("Density")
ax.set_title("Same expected profit, different risk", loc="left", fontsize=9.5)
ax.set_ylim(0, 8.5)
ax.legend(loc="upper left", fontsize=8)
save(fig, "fig5_wrong_vol")

Ns_w = np.array([16, 64, 256, 1024])
rows = []
for N in Ns_w:
    a = hedge_pnl(mw, N, M=M_SWEEP, sigma_hedge=sig_imp, seed=5)
    b = hedge_pnl(mw, N, M=M_SWEEP, sigma_hedge=sig_real, sigma_price=sig_imp, seed=5)
    rows.append((N, a.mean(), a.std(), b.mean(), b.std()))
table = "\n".join(f"{N} & {am:.3f} & {as_:.3f} & {bm:.3f} & {bs_:.3f} \\\\"
                  for N, am, as_, bm, bs_ in rows)
numbers["WrongVolRows"] = table

# ------------------------------------------------------------ write numbers
lines = ["% Generated by experiments/run_experiments.py -- do not edit by hand."]
lines += [f"\\newcommand{{\\{k_}}}{{{v}}}" for k_, v in numbers.items()]
(ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n")
print(f"done: {len(numbers)} numbers, figures in {FIG}")
