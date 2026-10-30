"""
Hedge with fixed learning rate vs AdaHedge on three synthetic regimes.

Run:  python main.py
Outputs (figures, tables, raw summaries) are written to results/.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from algorithms import Hedge, AdaHedge                           # noqa: E402
from losses import REGIMES                                       # noqa: E402
from experiments import (run_online, regret_curve, cumulative_loss,  # noqa: E402
                         best_expert_cumulative_loss, hedge_regret_bound,
                         adahedge_lemma4_bound)
from plots import plot_regime, plot_eta_sensitivity             # noqa: E402

# ----------------------------------------------------------------- settings
K = 10                     # number of experts
T = 10_000                 # number of rounds
N_REPS = 100               # independent repetitions per regime
PHI = 2.0                  # AdaHedge restart factor (as in the paper's experiments)
SEED = 2026                # master seed
ETA_T = np.sqrt(8 * np.log(K) / T)       # worst-case optimal tuning
ETA_MULTIPLIERS = [0.25, 1, 4, 16, 64]   # Hedge is run with eta = c * ETA_T
TITLES = {"stochastic": "Stochastic", "adversarial": "Adversarial",
          "low_gap": "Low-gap"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def hedge_name(c):
    return f"Hedge(eta={c:g}*eta_T)"


def run_regime(regime_id, regime):
    generate = REGIMES[regime]
    etas = {hedge_name(c): c * ETA_T for c in ETA_MULTIPLIERS}
    names = list(etas) + ["AdaHedge"]
    regret = {n: np.empty((N_REPS, T)) for n in names}
    cum = {n: np.empty((N_REPS, T)) for n in names}
    best_cum = np.empty((N_REPS, T))
    segments, final_eta, lemma4_ok, best_is_0 = [], [], [], []

    for r in range(N_REPS):
        rng = np.random.default_rng([SEED, regime_id, r])
        losses = generate(T, K, rng)
        best_cum[r] = best_expert_cumulative_loss(losses)
        # does the expert with the smallest mean loss win in hindsight?
        best_is_0.append(np.argmin(losses.sum(axis=0)) == 0)
        for n in names:
            alg = AdaHedge(K, phi=PHI) if n == "AdaHedge" else Hedge(K, etas[n])
            ll = run_online(alg, losses)
            regret[n][r] = regret_curve(ll, losses)
            cum[n][r] = cumulative_loss(ll)
            if n == "AdaHedge":
                segments.append(alg.n_segments)
                final_eta.append(alg.eta)
                lemma4_ok.append(regret[n][r, -1] <
                                 adahedge_lemma4_bound(alg.n_segments, K, PHI))
        print(f"  {regime}: repetition {r + 1}/{N_REPS} done", end="\r")
    print()

    z = 1.96 / np.sqrt(N_REPS)  # 95% normal confidence interval of the mean
    summary = {
        "mean_regret": {n: regret[n].mean(0) for n in names},
        "ci_regret": {n: z * regret[n].std(0, ddof=1) for n in names},
        "mean_cum_loss": {n: cum[n].mean(0) for n in names},
        "best_cum_loss": best_cum.mean(0),
        "final_mean": {n: regret[n][:, -1].mean() for n in names},
        "final_std": {n: regret[n][:, -1].std(ddof=1) for n in names},
        "final_ci": {n: z * regret[n][:, -1].std(ddof=1) for n in names},
        "ada_segments": np.array(segments),
        "ada_final_eta": np.array(final_eta),
        "ada_lemma4_ok": float(np.mean(lemma4_ok)),
        "best_expert_is_0": float(np.mean(best_is_0)),
        "best_final_loss": best_cum[:, -1].mean(),
        "log_regret": regime in ("stochastic", "adversarial"),
    }
    return summary, etas


def write_outputs(summaries, etas):
    os.makedirs(OUT, exist_ok=True)
    names = list(etas) + ["AdaHedge"]

    # CSV with the final regret of every algorithm in every regime
    with open(os.path.join(OUT, "final_regret.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "algorithm", "eta", "mean_final_regret",
                    "std_final_regret", "ci95_halfwidth"])
        for reg, s in summaries.items():
            for n in names:
                w.writerow([reg, n, etas.get(n, "adaptive"),
                            f"{s['final_mean'][n]:.3f}",
                            f"{s['final_std'][n]:.3f}",
                            f"{s['final_ci'][n]:.3f}"])

    # JSON with AdaHedge diagnostics and theoretical reference values
    diag = {"K": K, "T": T, "n_reps": N_REPS, "phi": PHI, "seed": SEED,
            "eta_T": ETA_T,
            "worst_case_bound_eta_T": float(np.sqrt(T * np.log(K) / 2)),
            "regimes": {}}
    for reg, s in summaries.items():
        diag["regimes"][reg] = {
            "adahedge_segments_mean": float(s["ada_segments"].mean()),
            "adahedge_segments_min": int(s["ada_segments"].min()),
            "adahedge_segments_max": int(s["ada_segments"].max()),
            "adahedge_final_eta_mean": float(s["ada_final_eta"].mean()),
            "adahedge_lemma4_bound_holds_fraction": s["ada_lemma4_ok"],
            # only meaningful where expert 0 has the smallest mean loss
            "fraction_runs_designated_expert_is_best":
                None if reg == "adversarial" else s["best_expert_is_0"],
            "best_expert_final_loss_mean": float(s["best_final_loss"]),
        }
    with open(os.path.join(OUT, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # LaTeX table used by the report
    regs = list(summaries)
    lines = [r"\begin{tabular}{l r " + " ".join(["r"] * len(regs)) + "}",
             r"\toprule",
             r"Algorithm & $\eta$ & " + " & ".join(TITLES[r] for r in regs)
             + r" \\", r"\midrule"]
    for n in names:
        if n == "AdaHedge":
            lines.append(r"\midrule")
            label, eta_str = "AdaHedge ($\\phi=2$)", "adaptive"
        else:
            c = etas[n] / ETA_T
            c_str = "1/4" if np.isclose(c, 0.25) else f"{c:g}"
            label = rf"Hedge, $\eta={c_str}\,\eta_T$"
            eta_str = f"{etas[n]:.3f}"
        cells = []
        for r in regs:
            s = summaries[r]
            best = min(s["final_mean"].values())
            val = rf"{s['final_mean'][n]:.1f} $\pm$ {s['final_std'][n]:.1f}"
            if np.isclose(s["final_mean"][n], best):
                val = r"\textbf{" + val + "}"
            cells.append(val)
        lines.append(f"{label} & {eta_str} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "table_final_regret.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # figures
    for reg, s in summaries.items():
        plot_regime(s, reg, TITLES[reg], etas, ETA_T,
                    os.path.join(OUT, f"{reg}.pdf"))
    plot_eta_sensitivity(summaries, TITLES, etas, ETA_T, T, K,
                         os.path.join(OUT, "eta_sensitivity.pdf"))


def main():
    print(f"K={K}, T={T}, repetitions={N_REPS}, eta_T={ETA_T:.4f}")
    summaries, etas = {}, None
    for regime_id, regime in enumerate(["stochastic", "adversarial", "low_gap"]):
        summaries[regime], etas = run_regime(regime_id, regime)
    write_outputs(summaries, etas)
    print(f"Results written to {OUT}")


if __name__ == "__main__":
    main()
