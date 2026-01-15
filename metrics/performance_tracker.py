import time


class PerformanceTracker:
    """
    Tracks performance metrics for speculative decoding.
    Passive observer only — no side effects.
    """

    def __init__(self):
        self.reset()

    # --------------------------------------------------
    # Core lifecycle
    # --------------------------------------------------

    def reset(self):
        self.start_time = None
        self.end_time = None

        self.tokens_generated = 0
        self.target_forward_calls = 0
        self.draft_forward_calls = 0

    # --------------------------------------------------
    # Timing
    # --------------------------------------------------

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    # --------------------------------------------------
    # Counters
    # --------------------------------------------------

    def record_tokens(self, n: int = 1):
        self.tokens_generated += n

    def record_target_forward(self, n: int = 1):
        self.target_forward_calls += n

    def record_draft_forward(self, n: int = 1):
        self.draft_forward_calls += n

    # --------------------------------------------------
    # Derived metrics
    # --------------------------------------------------

    @property
    def total_time_sec(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def latency_per_token_ms(self) -> float:
        if self.tokens_generated == 0:
            return 0.0
        return (self.total_time_sec / self.tokens_generated) * 1000

    @property
    def throughput_tokens_per_sec(self) -> float:
        if self.total_time_sec == 0:
            return 0.0
        return self.tokens_generated / self.total_time_sec

    # --------------------------------------------------
    # Reporting
    # --------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_time_sec": round(self.total_time_sec, 4),
            "tokens_generated": self.tokens_generated,
            "latency_per_token_ms": round(self.latency_per_token_ms, 3),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
            "target_forward_calls": self.target_forward_calls,
            "draft_forward_calls": self.draft_forward_calls,
        }

    def speedup_vs_baseline(self, baseline_latency_per_token_ms: float) -> float:
        """
        Compute speedup relative to baseline decoding.
        """
        if baseline_latency_per_token_ms <= 0:
            return 0.0
        return baseline_latency_per_token_ms / self.latency_per_token_ms
