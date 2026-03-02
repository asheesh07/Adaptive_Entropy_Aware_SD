import torch
import torch.nn.functional as F

class RejectionSampler:
    def __init__(self, target_model):
        self.target_model = target_model

    def handle(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> torch.Tensor:

        # Use unified distribution builders
        t_probs, _, _ = self.target_model.build_distribution(target_logits)
        d_probs, _, _ = self.target_model.build_distribution(draft_logits)

        # Ensure same device
        d_probs = d_probs.to(t_probs.device)

        corrected = torch.clamp(t_probs - d_probs, min=0.0)
        corrected_sum = corrected.sum(dim=-1, keepdim=True)

        if corrected_sum.item() < 1e-8:
            corrected = t_probs
        else:
            corrected = corrected / corrected_sum

        return torch.multinomial(corrected, num_samples=1)

    def handle_bonus(self, target_logits: torch.Tensor):
        probs, _, _ = self.target_model.build_distribution(target_logits)
        return torch.multinomial(probs, num_samples=1)