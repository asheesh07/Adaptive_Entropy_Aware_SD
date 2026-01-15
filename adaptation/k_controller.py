class KController:
    def __init__(self,entropy_threshold,k_values,k_max,acceptance_min=0,acceptance_max=None):
        assert len(k_values) == len(entropy_threshold) + 1, "k_values should have one more element than entropy_threshold"
        self.entropy_threshold = entropy_threshold
        self.k_values = k_values
        self.k_max = k_max
        self.acceptance_min = acceptance_min
        self.acceptance_max = acceptance_max
        
    def entropy_to_k(self,entropy):
        for threshold,k in zip(self.entropy_threshold,self.k_values):
            if entropy < threshold:
                return k
        return self.k_values[-1]
    
    def _acceptance_feedback(self,k,acceptance_rate):
        if acceptance_rate < self.acceptance_min:
            k=max(0,k//2)
        elif acceptance_rate > self.acceptance_max:
            k=min(self.k_max,k+1)
        return k
    
    def decide__k(self,entropy,acceptance_rate=None):
        k=self.entropy_to_k(entropy)
        k=self._acceptance_feedback(k,acceptance_rate)
        
        return max(0,min(k,self.k_max))
    