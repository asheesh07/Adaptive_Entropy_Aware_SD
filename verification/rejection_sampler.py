import torch
import torch.nn.functional as F

class RejectionSampler:
    def __init__(self, target_model):
        self.target_model = target_model

    def handle(
        self,
        target_logits: torch.Tensor, 
        draft_logits: torch.Tensor,   
        temperature: float = 1.0,
    ) -> torch.Tensor:
        t_probs = F.softmax(target_logits / max(temperature, 1e-5), dim=-1)  
        d_probs = F.softmax(draft_logits.to(target_logits.device) / max(temperature, 1e-5), dim=-1)  

        corrected = torch.clamp(t_probs - d_probs, min=0.0)
        corrected_sum = corrected.sum(dim=-1, keepdim=True)

        if corrected_sum.item() < 1e-8:
            corrected = t_probs
        else:
            corrected = corrected / corrected_sum

        next_token = torch.multinomial(corrected, num_samples=1)  
        return next_token

    def handle_bonus(
        self,
        target_logits: torch.Tensor,  
        temperature: float = 1.0,
    ) -> torch.Tensor:

        probs = F.softmax(target_logits / max(temperature, 1e-5), dim=-1)  
        next_token = torch.multinomial(probs, num_samples=1) 
        return next_token