# Hedge vs AdaHedge — Online Learning with Expert Advice

Experimental project for *Statistical Methods for Machine Learning* (Assignment 1 — Online Learning, Adaptive Hedge).

Hedge with a fixed learning rate and AdaHedge (van Erven, Grünwald, Koolen, de Rooij, NeurIPS 2011) are implemented from scratch and compared on three synthetic loss regimes:

- **Stochastic**: one expert Bernoulli(0.3), the others Bernoulli(0.5);
- **Adversarial**: the best expert changes over time (the current leader always receives loss 1, the others Bernoulli(0.5));
- **Low-gap**: best expert Bernoulli(0.49), the others Bernoulli(0.5).

Hedge is run with η = c·η_T, c ∈ {1/4, 1, 4, 16, 64}, where η_T = √(8 ln K / T) is the worst-case tuning. Setting: K = 10 experts, T = 10 000 rounds, 100 repetitions per regime.


## Repository structure

```
main.py                  runs all the experiments and writes results/
src/algorithms.py        Hedge (fixed learning rate) and AdaHedge
src/losses.py            generators of the three loss regimes
src/experiments.py       online protocol, regret, theoretical bounds
src/plots.py             figures
results/                 figures (PDF), final_regret.csv, diagnostics.json, LaTeX table
```

## Reproducing the results

Requires Python 3 with `numpy` and `matplotlib` (used only for array arithmetic and plotting).

```bash
pip install -r requirements.txt
python main.py
```

All random sequences are generated from fixed seeds, so the run reproduces exactly the numbers in the report. Outputs:

- `results/stochastic.pdf`, `results/adversarial.pdf`, `results/low_gap.pdf`: cumulative loss and regret vs time;
- `results/eta_sensitivity.pdf`: final regret of Hedge as a function of η, compared with AdaHedge;
- `results/final_regret.csv`: final regret (mean, standard deviation, 95% CI) of every algorithm in every regime;
- `results/diagnostics.json`: AdaHedge segments, final learning rates and theoretical reference values;
- `results/table_final_regret.tex`: table included in the report.


## References

- T. van Erven, P. D. Grünwald, W. M. Koolen, S. de Rooij. *Adaptive Hedge*. NeurIPS 2011 (arXiv:1110.6416).
- T. R. Cesari, N. Cesa-Bianchi. *Hedge and Exp3 for sequential decision-making*. Lecture notes, 2025.
- Y. Freund, R. E. Schapire. *A decision-theoretic generalization of on-line learning and an
application to boosting*. Journal of Computer and System Sciences, 55:119–139, 1997.
- N. Cesa-Bianchi, G. Lugosi. *Prediction, Learning, and Games*. Cambridge University Press, 2006.