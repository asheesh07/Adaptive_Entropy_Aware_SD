import matplotlib.pyplot as plt


class MetricsVisualizer:
    """
    Visualization utilities for speculative decoding metrics.
    """

    # --------------------------------------------------
    # Acceptance & k behavior
    # --------------------------------------------------

    @staticmethod
    def plot_acceptance_over_time(acceptance_history):
        plt.figure(figsize=(8, 4))
        plt.plot(acceptance_history, label="Step Acceptance")
        plt.xlabel("Decoding Step")
        plt.ylabel("Acceptance Rate")
        plt.title("Acceptance Rate Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_k_over_time(k_history):
        plt.figure(figsize=(8, 4))
        plt.plot(k_history, label="Speculation Depth (k)")
        plt.xlabel("Decoding Step")
        plt.ylabel("k")
        plt.title("Adaptive k Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # Performance comparison
    # --------------------------------------------------

    @staticmethod
    def plot_latency_comparison(spec_latency_ms, baseline_latency_ms):
        labels = ["Baseline", "Speculative"]
        values = [baseline_latency_ms, spec_latency_ms]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.ylabel("Latency per Token (ms)")
        plt.title("Latency Comparison")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # Speculation efficiency
    # --------------------------------------------------

    @staticmethod
    def plot_speculation_efficiency(accepted, wasted):
        labels = ["Accepted", "Wasted"]
        values = [accepted, wasted]

        plt.figure(figsize=(6, 4))
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title("Speculation Efficiency")
        plt.tight_layout()
        plt.show()
