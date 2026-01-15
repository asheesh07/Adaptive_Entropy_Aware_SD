import torch

class DraftGenerationLoop:
    def __init__(self, draft_model):
        self.draft_model = draft_model

    @torch.no_grad()
    def generate(self, k: int) -> torch.Tensor:
        tokens = []

        for _ in range(k):
            logits = self.draft_model.forward_next(
                self.draft_model.last_token
            )
            token = self.draft_model.sample_token(logits)
            tokens.append(token)

        # 🔑 Convert list → tensor [1, k]
        return torch.cat(tokens, dim=1)
