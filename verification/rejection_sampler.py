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
        # Align vocab sizes by padding smaller one
        t_vocab = target_logits.shape[-1]
        d_vocab = draft_logits.shape[-1]
        
        if d_vocab < t_vocab:
            draft_logits = F.pad(draft_logits, (0, t_vocab - d_vocab), value=float('-inf'))
        elif t_vocab < d_vocab:
            target_logits = F.pad(target_logits, (0, d_vocab - t_vocab), value=float('-inf'))

        t_probs = F.softmax(target_logits / max(temperature, 1e-5), dim=-1)
        d_probs = F.softmax(draft_logits.to(target_logits.device) / max(temperature, 1e-5), dim=-1)

        corrected = torch.clamp(t_probs - d_probs, min=0.0)
        corrected_sum = corrected.sum(dim=-1, keepdim=True)

        if corrected_sum.item() < 1e-8:
            corrected = t_probs
        else:
            corrected = corrected / corrected_sum

        return torch.multinomial(corrected, num_samples=1)

    def handle_bonus(
        self,
        target_logits: torch.Tensor,  
        temperature: float = 1.0,
    ) -> torch.Tensor:

        probs = F.softmax(target_logits / max(temperature, 1e-5), dim=-1)  
        next_token = torch.multinomial(probs, num_samples=1) 
        return next_token