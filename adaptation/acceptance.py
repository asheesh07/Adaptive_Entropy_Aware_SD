class AcceptanceTracker:
    def __init__(self,alpha,initial_values):
        self.alpha=alpha
        self.value=initial_values
        
    def update(self,accepted_tokens,k):
        if k<=0:
            return self.value
        acceptance_rate= accepted_tokens / k
        self.value= self.alpha * self.value + (1 - self.alpha) * acceptance_rate
        
        return self.value
    