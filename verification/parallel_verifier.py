class ParallelVerifier:
    def __init__(self,target_model):
        self.target_model=target_model
        
    def verify(self,draft_tokens):
        k= draft_tokens.shape[1]
        if k ==0:
            return 0,self.target_model.kv_cache
        outputs =self.target_model.model(input_ids=draft_tokens.to(self.target_model.device), past_key_values=self.target_model.kv_cache, use_cache=True,return_dict=True)
        logits= outputs.logits
        new_kv_cache= outputs.past_key_values
        accepted_tokens =0
        for i in range(draft_tokens.shape[1]):
            target_token = self.target_model.sample_token(logits[:,i,:])
            if target_token.item() == draft_tokens[0,i].item():
                accepted_tokens += 1
            else:
                break
        return accepted_tokens,new_kv_cache