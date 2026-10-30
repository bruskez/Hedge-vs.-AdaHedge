"""Plotting utilities (matplotlib is used only to draw the results)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import hedge_regret_bound

ADA_COLOR = "#eb6834"      # AdaHedge: a distinct categorical hue
BEST_COLOR = "#0b0b0b"     # best expert in hindsight
TEXT_MUTED = "#52514e"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,          # embed TrueType fonts in the PDF figures
})


def hedge_colors(n):
    """Sequential blue ramp: lighter = smaller eta, darker = larger eta."""
    return [plt.cm.Blues(x) for x in np.linspace(0.35, 1.0, n)]


def _labels(names, etas, eta_T):
    out = []
    for name in names:
        if name == "AdaHedge":
            out.append("AdaHedge")
        else:
            c = etas[name] / eta_T
            c_str = "1/4" if np.isclose(c, 0.25) else f"{c:g}"
            out.append(rf"Hedge $\eta={c_str}\,\eta_T$ ({etas[name]:.3f})")
    return out


def plot_regime(summary, regime, title, etas, eta_T, path):
    """Two panels: cumulative loss vs time and regret vs time."""
    names = list(summary["mean_regret"].keys())
    hedge_names = [n for n in names if n != "AdaHedge"]
    colors = dict(zip(hedge_names, hedge_colors(len(hedge_names))))
    colors["AdaHedge"] = ADA_COLOR
    labels = dict(zip(names, _labels(names, etas, eta_T)))
    T = len(summary["best_cum_loss"])
    t = np.arange(1, T + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # --- cumulative loss ---
    for n in names:
        ax1.plot(t, summary["mean_cum_loss"][n], color=colors[n],
                 label=labels[n])
    ax1.plot(t, summary["best_cum_loss"], color=BEST_COLOR, ls="--", lw=1.2,
             label=r"best expert $L^*_t$")
    ax1.set_xlabel("round $t$")
    ax1.set_ylabel("cumulative loss")
    ax1.set_title(f"{title}: cumulative loss")

    # --- regret ---
    for n in names:
        m = summary["mean_regret"][n]
        ci = summary["ci_regret"][n]
        ax2.plot(t, m, color=colors[n], label=labels[n])
        ax2.fill_between(t, m - ci, m + ci, color=colors[n], alpha=0.18,
                         lw=0)
    ax2.axhline(0, color=TEXT_MUTED, lw=0.6)
    ax2.set_xlabel("round $t$")
    ax2.set_ylabel(r"regret $R(t)$")
    ax2.set_title(f"{title}: regret vs time")
    if summary.get("log_regret", False):
        # symmetric-log scale: linear around 0, logarithmic above 1
        ax2.set_yscale("symlog", linthresh=1)
        ax2.set_ylim(bottom=0)
    # one shared legend below the two panels
    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(path)
    plt.close(fig)


def plot_eta_sensitivity(summaries, titles, etas, eta_T, T, K, path):
    """Final regret of Hedge as a function of eta, with AdaHedge as reference
    and the worst-case bound ln K / eta + eta T / 8."""
    fig, axes = plt.subplots(1, len(summaries), figsize=(7.2, 2.7))
    grid = np.logspace(np.log10(min(etas.values()) / 1.5),
                       np.log10(max(etas.values()) * 1.5), 200)
    for ax, (regime, s) in zip(axes, summaries.items()):
        hedge_names = [n for n in s["final_mean"] if n != "AdaHedge"]
        x = np.array([etas[n] for n in hedge_names])
        y = np.array([s["final_mean"][n] for n in hedge_names])
        e = np.array([s["final_ci"][n] for n in hedge_names])
        ax.plot(grid, hedge_regret_bound(grid, T, K), color=TEXT_MUTED,
                ls=":", lw=1.2, label=r"bound $\ln K/\eta+\eta T/8$")
        ax.errorbar(x, y, yerr=e, color=plt.cm.Blues(0.85), marker="o",
                    ms=5, capsize=3, label="Hedge (fixed $\\eta$)")
        a = s["final_mean"]["AdaHedge"]
        ae = s["final_ci"]["AdaHedge"]
        ax.axhline(a, color=ADA_COLOR, lw=1.4, label="AdaHedge")
        ax.axhspan(a - ae, a + ae, color=ADA_COLOR, alpha=0.18, lw=0)
        ax.axvline(eta_T, color=TEXT_MUTED, lw=0.6, ls="--")
        ax.text(eta_T, 0.97, r" $\eta_T$", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color=TEXT_MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1, 1e4)
        ax.set_xlabel(r"learning rate $\eta$")
        ax.set_ylabel(r"final regret $R(T)$")
        ax.set_title(titles[regime])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
