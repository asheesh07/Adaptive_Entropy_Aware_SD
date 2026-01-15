import torch
class EntropyCalculator:
    @torch.no_grad()
    def compute(self,logits: torch.Tensor) -> torch.Tensor:
        
        log_probs = torch.log_softmax(logits, dim=-1)
        probs= torch.exp(log_probs)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=float("inf"), posinf=float("inf"))

        return entropy.item()