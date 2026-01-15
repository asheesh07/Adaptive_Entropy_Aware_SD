class ThresholdAdjuster:
    def __init__(self,entropy_bins,min_val,max_val,adjustment_rate,target_acceptance):
        self.entropy_bins=entropy_bins
        self.min_val=min_val
        self.max_val=max_val
        self.adjustment_rate=adjustment_rate
        self.target_acceptance=target_acceptance