"""
Synthetic loss sequences for the three regimes of the assignment.

Each generator returns a (T, K) array whose row t is the loss vector l_t in
{0,1}^K. All randomness comes from the numpy Generator passed as argument, so
every sequence is reproducible from its seed. The sequences do not depend on
the learner (oblivious adversary), so all algorithms are evaluated on exactly
the same losses within a repetition.
"""
import numpy as np


def stochastic_losses(T, K, rng, p_best=0.3, p_other=0.5):
    """I.i.d. Bernoulli losses: expert 0 ~ Bernoulli(p_best),
    experts 1..K-1 ~ Bernoulli(p_other)."""
    p = np.full(K, p_other)
    p[0] = p_best
    return (rng.random((T, K)) < p).astype(float)


def low_gap_losses(T, K, rng):
    """Same as the stochastic regime, but with a tiny gap:
    expert 0 ~ Bernoulli(0.49), the others ~ Bernoulli(0.5)."""
    return stochastic_losses(T, K, rng, p_best=0.49, p_other=0.5)


def adversarial_losses(T, K, rng):
    """Adversarial regime in which the best expert keeps changing.

    At every round the adversary looks at the cumulative losses of the experts
    (not at the learner) and gives loss 1 to the current leader, i.e. the
    expert with the smallest cumulative loss (ties broken uniformly at random).
    All other experts receive Bernoulli(1/2) losses. Hence whichever expert
    is currently best is immediately punished and the identity of the best
    expert changes continually. This is the K-expert, randomised version of
    the classical counter-example for Follow-the-Leader (lecture notes, and
    Section 1 of the AdaHedge paper), in which the leader is always wrong.
    """
    losses = np.empty((T, K))
    cum = np.zeros(K)
    for t in range(T):
        l = (rng.random(K) < 0.5).astype(float)
        leaders = np.flatnonzero(cum == cum.min())
        l[rng.choice(leaders)] = 1.0
        losses[t] = l
        cum += l
    return losses


REGIMES = {
    "stochastic": stochastic_losses,
    "adversarial": adversarial_losses,
    "low_gap": low_gap_losses,
}
