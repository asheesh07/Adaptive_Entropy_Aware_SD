import torch

class DraftGenerationLoop:
    def __init__(self, draft_model):
        self.draft_model = draft_model

    @torch.no_grad()
    def generate(self, k: int, last_token: torch.Tensor) -> torch.Tensor:
        if k == 0:
            return torch.empty(
                (1, 0),
                dtype=torch.long,
                device=self.draft_model.device,
            )
        tokens = []
        current_token = last_token
        for _ in range(k):
            logits = self.draft_model.forward_next(
                current_token
            )
            next_token = self.draft_model.sample_token(logits)
            tokens.append(next_token)
            current_token = next_token 

        # 🔑 Convert list → tensor [1, k]
        return torch.cat(tokens, dim=1)
