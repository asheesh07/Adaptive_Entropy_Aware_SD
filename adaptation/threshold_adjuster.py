class ThresholdAdjuster:
    def __init__(self,entropy_bins,min_scale: float = 0.7,
        max_scale: float = 1.3,
        adjust_rate: float = 0.05,
        target_acceptance: float = 0.75,):
        self.abs_min: float = 0.1,
        self.abs_max: float = 10.0
        self.entropy_bins = [float(b) for b in entropy_bins]
        self.min_scale=min_scale
        self.max_scale=max_scale
        self.adjust_rate=adjust_rate
        self.target_acceptance=target_acceptance
    def update(self, acceptance_rate: float):
        """
        Adjust entropy thresholds based on acceptance feedback.
        """
        # Positive if acceptance is better than target
        error = acceptance_rate - self.target_acceptance

        # Compute scaling factor
        scale = 1.0 + self.adjust_rate * error
        scale = max(self.min_scale, min(scale, self.max_scale))

        # Apply scaling
        self.entropy_bins = [max(self.abs_min, min(b * scale, self.abs_max)) 
    for b in self.entropy_bins]

        return self.entropy_bins