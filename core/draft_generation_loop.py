import torch

class DraftGenerationLoop:
    def __init__(self, draft_model):
        self.draft_model = draft_model

    @torch.no_grad()
    def generate(self, k: int, last_token: torch.Tensor):
        if k == 0:
            return (
                torch.empty((1, 0), dtype=torch.long, device=self.draft_model.device),
                [],
                [],
            )

        tokens = []
        entropies = []
        probs_list = []

        current_token = last_token

        for _ in range(k):
            logits = self.draft_model.forward_next(current_token)
            probs, entropy, entropy_norm = self.draft_model.build_distribution(logits)

            # Sample
            next_token = self.draft_model.sample_from_probs(probs)

            tokens.append(next_token)
            entropies.append(entropy)
            probs_list.append(probs)

            current_token = next_token

        return torch.cat(tokens, dim=1), entropies, probs_list