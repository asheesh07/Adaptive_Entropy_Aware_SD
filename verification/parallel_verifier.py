import torch
class ParallelVerifier:
    def __init__(self, target_model, max_rank=3):
        self.target_model = target_model
        self.max_rank = max_rank

    @torch.no_grad()
    def verify(self, draft_tokens):
        assert draft_tokens.dim() == 2 and draft_tokens.size(0) == 1

        k = draft_tokens.shape[1]
        if k == 0:
            return 0, self.target_model.kv_cache

        accepted = 0

        # Save original cache pointer
        original_cache = self.target_model.kv_cache

        for i in range(k):
            token = draft_tokens[:, i:i+1]  # shape [1,1]

            logits = self.target_model.forward_next(token)
            probs = F.softmax(logits[0], dim=-1)

            topk = torch.topk(probs, self.max_rank).indices

            if token.item() in topk:
                accepted += 1
            else:
                break

        # The cache now contains only the accepted prefix
        temp_cache = self.target_model.kv_cache

        # Restore original cache (engine will decide what to commit)
        self.target_model.kv_cache = original_cache

        return accepted, temp_cache
