"""
Online evaluation protocol and performance measures.

Online learning has no train/test split: the learner is evaluated on the same
sequence it learns from, but in round t it only uses the losses l_1..l_{t-1}
(it commits to w_t before l_t is revealed). Performance is measured by the
regret with respect to the best expert in hindsight,

    R(t) = sum_{s<=t} w_s . l_s  -  min_k sum_{s<=t} l_s(k).
"""
import numpy as np


def run_online(algorithm, losses):
    """Play the DTOL protocol on a (T, K) loss matrix.

    Returns the array of per-round learner losses w_t . l_t.
    """
    T = losses.shape[0]
    learner_loss = np.empty(T)
    for t in range(T):
        w = algorithm.weights()          # decision made before seeing l_t
        learner_loss[t] = w @ losses[t]  # expected loss of the learner
        algorithm.update(losses[t])      # l_t is revealed
    return learner_loss


def cumulative_loss(learner_loss):
    """Cumulative loss of the learner, sum_{s<=t} w_s . l_s, for every t."""
    return np.cumsum(learner_loss)


def best_expert_cumulative_loss(losses):
    """L*_t = min_k L_t(k) for every t (the best expert may change with t)."""
    return np.cumsum(losses, axis=0).min(axis=1)


def regret_curve(learner_loss, losses):
    """R(t) for t = 1..T."""
    return cumulative_loss(learner_loss) - best_expert_cumulative_loss(losses)


def hedge_regret_bound(eta, T, K):
    """Worst-case bound of Hedge with fixed eta: ln K / eta + eta T / 8
    (from delta_t <= eta/8, Lemma 1 of the AdaHedge paper)."""
    return np.log(K) / eta + eta * T / 8.0


def adahedge_lemma4_bound(m, K, phi):
    """Regret bound of AdaHedge after starting m segments (Lemma 4)."""
    return (2 * np.log(K) * (phi ** m - 1) / (phi - 1)
            + m * (np.log(K) / (np.e - 1) + 1 / 8))
