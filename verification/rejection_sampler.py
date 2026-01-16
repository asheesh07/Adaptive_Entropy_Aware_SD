import torch
class RejectionSampler:
    def __init__(self,target_model,draft_model):
        self.target_model=target_model
        self.draft_model=draft_model
        
    def handle(self,last_committed_token: torch.Tensor,):
        
        outputs = self.target_model.model(
            input_ids=last_committed_token.to(self.target_model.device),
            past_key_values=self.target_model.kv_cache,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        self.target_model.kv_cache = outputs.past_key_values
        self.target_model.position += 1

        self.draft_model.kv_cache = self.target_model.kv_cache
        self.draft_model.position = self.target_model.position


        return next_token

    def _slice_kv_cache(self, kv_cache, accepted_tokens):
        new_cache = []
        for k,v in kv_cache:
            new_k = k[:, :, :accepted_tokens, :].contiguous()
            new_v = v[:, :, :accepted_tokens, :].contiguous()
            new_cache.append((new_k, new_v))
        return tuple(new_cache)