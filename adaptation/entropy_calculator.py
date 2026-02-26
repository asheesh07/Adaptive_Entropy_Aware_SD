import torch
class EntropyCalculator:
    
    @torch.no_grad()
    def compute(self,logits: torch.Tensor) -> torch.Tensor:
        
        log_probs = torch.log_softmax(logits, dim=-1)
        probs= torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0)

        return entropy