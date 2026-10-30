"""
Hedge (fixed learning rate) and AdaHedge, implemented from scratch.

Both algorithms follow the decision-theoretic online learning (DTOL) protocol:
at every round t the learner first outputs a probability vector w_t over the K
experts (method `weights`), then the loss vector l_t in [0,1]^K is revealed
(method `update`). The loss of the learner in round t is the expected loss
w_t . l_t. The learner never sees l_t before committing to w_t.

All computations are done in the log domain to avoid numerical under/overflow:
the weights are always obtained as a softmax of -eta * (cumulative losses).
"""
import numpy as np


def _log_sum_exp(a):
    """Numerically stable log(sum(exp(a)))."""
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def _softmax_from_logits(logits):
    """Probability vector proportional to exp(logits)."""
    z = logits - np.max(logits)
    p = np.exp(z)
    return p / p.sum()


class Hedge:
    """Hedge (exponential weights) with a fixed learning rate eta.

    w_{t+1}(k) is proportional to exp(-eta * L_t(k)), where L_t(k) is the
    cumulative loss of expert k after t rounds; w_1 is uniform.
    """

    def __init__(self, n_experts, eta):
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.K = n_experts
        self.eta = eta
        self.cum_loss = np.zeros(n_experts)   # L_{t-1}(k)

    def weights(self):
        return _softmax_from_logits(-self.eta * self.cum_loss)

    def update(self, loss):
        self.cum_loss += loss


class AdaHedge:
    """AdaHedge(phi), Algorithm 1 of van Erven, Grunwald, Koolen, de Rooij (2011).

    Hedge is run in segments. Segment i uses the learning rate
    eta_i = phi^(1-i) (so eta_1 = 1) and starts from uniform weights. Inside a
    segment the cumulative mixability gap

        Delta = sum_t [ w_t . l_t + (1/eta) ln(w_t . exp(-eta l_t)) ]

    is monitored; as soon as it reaches the budget
    b(eta) = (1/eta + 1/(e-1)) ln K, a new segment is started with
    eta <- eta / phi.
    """

    def __init__(self, n_experts, phi=2.0):
        if phi <= 1:
            raise ValueError("phi must be > 1")
        self.K = n_experts
        self.phi = phi
        self.eta = phi            # divided by phi when the first segment starts
        self.n_segments = 0
        self.segment_starts = []  # rounds (1-based) at which segments start
        self.t = 0
        self._new_segment_pending = True

    def _start_segment(self):
        self.eta = self.eta / self.phi
        self.budget = (1.0 / (np.e - 1.0) + 1.0 / self.eta) * np.log(self.K)
        self.delta = 0.0
        self.seg_cum_loss = np.zeros(self.K)  # losses inside the segment only
        self.n_segments += 1
        self.segment_starts.append(self.t + 1)

    def weights(self):
        # "if t = 1 or Delta >= b then start a new segment"
        if self._new_segment_pending:
            self._start_segment()
            self._new_segment_pending = False
        return _softmax_from_logits(-self.eta * self.seg_cum_loss)

    def update(self, loss):
        w = self.weights()
        log_w = np.log(np.maximum(w, 1e-300))
        hedge_loss = float(w @ loss)
        # mix loss: -(1/eta) ln( w . exp(-eta l) ), computed in the log domain
        mix_loss = -_log_sum_exp(log_w - self.eta * loss) / self.eta
        # mixability gap delta_t >= 0 (clip tiny negative rounding errors)
        self.delta += max(hedge_loss - mix_loss, 0.0)
        self.seg_cum_loss += loss
        self.t += 1
        if self.delta >= self.budget:
            self._new_segment_pending = True
