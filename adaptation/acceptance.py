class AcceptanceTracker:
    def __init__(self, alpha: float = 0.1, initial_value: float = 0.7):
        self.alpha=alpha
        self.value=initial_value

    def update(self, accepted_tokens, k):
        if k<=0:
            return self.value
        acceptance_rate= accepted_tokens / k
        self.value= (1-self.alpha) * self.value + self.alpha * acceptance_rate
        
        return self.value
    