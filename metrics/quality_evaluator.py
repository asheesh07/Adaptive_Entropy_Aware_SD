from collections import defaultdict


class QualityEvaluator:
    """
    Tracks speculative decoding quality metrics.
    Passive observer only.
    """

    def __init__(self):
        self.reset()

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def reset(self):
        self.total_steps = 0

        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0

        self.rejection_steps = 0

        self.k_history = []
        self.acceptance_history = []

    # --------------------------------------------------
    # Recording
    # --------------------------------------------------

    def record_step(self, k: int, accepted_tokens: int):
        """
        Record one speculative decoding step.

        Args:
            k: speculation depth used
            accepted_tokens: number of tokens accepted
        """
        self.total_steps += 1

        self.total_draft_tokens += k
        self.total_accepted_tokens += accepted_tokens

        if accepted_tokens < k:
            self.rejection_steps += 1

        self.k_history.append(k)

        step_acceptance = accepted_tokens / k if k > 0 else 1.0
        self.acceptance_history.append(step_acceptance)

    # --------------------------------------------------
    # Derived metrics
    # --------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def rejection_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.rejection_steps / self.total_steps

    @property
    def average_k(self) -> float:
        if not self.k_history:
            return 0.0
        return sum(self.k_history) / len(self.k_history)

    @property
    def wasted_speculation(self) -> float:
        """
        Fraction of speculative tokens that were rejected.
        """
        if self.total_draft_tokens == 0:
            return 0.0
        wasted = self.total_draft_tokens - self.total_accepted_tokens
        return wasted / self.total_draft_tokens

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "average_k": round(self.average_k, 2),
            "wasted_speculation": round(self.wasted_speculation, 4),
        }
