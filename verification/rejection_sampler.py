import torch
class RejectionSampler:
    def __init__(self,target_model,draft_model):
        self.target_model=target_model
        self.draft_model=draft_model
        
    def handle(self,accepted_tokens,temp_target_kv_cache,last_committed_token: torch.Tensor,):
        
        if accepted_tokens > 0:
            self.target_model.kv_cache = self._slice_kv_cache(
                temp_target_kv_cache,
                accepted_tokens,
            )
            self.target_model.position += accepted_tokens

        logits = self.target_model.forward_next(last_committed_token)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token

    def _slice_kv_cache(self, kv_cache, accepted_tokens):
        new_cache = []
        for k,v in kv_cache:
            new_k = k[:, :, :accepted_tokens, :].contiguous()
            new_v = v[:, :, :accepted_tokens, :].contiguous()
            new_cache.append((new_k, new_v))
        return tuple(new_cache)