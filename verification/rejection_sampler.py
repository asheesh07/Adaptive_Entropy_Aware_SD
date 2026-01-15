class RejectionSampler:
    def __init__(self,target_model,draft_model):
        self.target_model=target_model
        self.draft_model=draft_model
        
    def handle_token(self,accepted_tokens):
        
        if accepted_tokens > 0:
            self.target_model.kv_cache = self._slice_kv_cache(self.target_model.kv_cache, accepted_tokens)
            self.target_model.position += accepted_tokens
            
            self.draft_model.reset_cache()
            self.draft_model.kv_cache =self.target_model.kv_cache
            self.draft_model.position = self.target_model.position
            
            logits = self.target_model.forward_next(None)
            next_token = self.target_model.sample_token(logits)
            
            self.draft_model.forward_next(next_token)
            self.target_model.forward_verify(next_token)
            
        return next_token

    def _slice_kv_cache(self, kv_cache, accepted_tokens):
        new_cache = []
        for k,v in kv_cache:
            new_k = k[:, :, : , :accepted_tokens, :].contiguous()
            new_v = v[:, :, : , :accepted_tokens, :].contiguous()
            new_cache.append((new_k, new_v))
        return tuple(new_cache)