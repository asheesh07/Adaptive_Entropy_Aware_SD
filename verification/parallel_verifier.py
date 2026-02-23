import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

class ParallelVerifier:
    def __init__(self, target_model):
        self.target_model = target_model

    @torch.no_grad()
    def verify(self, draft_tokens):
        assert draft_tokens.dim() == 2 and draft_tokens.size(0) == 1

        k = draft_tokens.shape[1]
        if k == 0:
            return 0, self.target_model.kv_cache

        position_before = self.target_model.position

        accepted = 0

        for i in range(k):
            token = draft_tokens[:, i:i+1]

            logits = self.target_model.forward_next(token)
            probs = F.softmax(logits[0], dim=-1)

            draft_token_id = token.item()
            token_prob = probs[draft_token_id].item()

            u = torch.rand(1).item()
            if u < token_prob:
                accepted += 1
            else:
                break

        temp_cache = self.target_model.kv_cache

        self.target_model.rollback_kv_cache(position_before)

        return accepted, temp_cache