import torch
import torch.nn.functional as F
class ParallelVerifier:
    def __init__(self,target_model,max_rank =3):
        self.target_model=target_model
        self.max_rank = max_rank
    
    @torch.no_grad()
    def verify(self,draft_tokens):
        assert draft_tokens.dim() == 2 and draft_tokens.size(0) == 1

        k= draft_tokens.shape[1]
        if k ==0:
            return 0,self.target_model.kv_cache
        outputs =self.target_model.model(input_ids=draft_tokens.to(self.target_model.device), past_key_values=self.target_model.kv_cache, use_cache=True,return_dict=True)
        logits= outputs.logits
        new_kv_cache= outputs.past_key_values
        accepted_tokens =0
        for i in range(draft_tokens.shape[1]):
            token_id = draft_tokens[0,i]
            probs = F.softmax(logits[0, i], dim=-1)
            topk = torch.topk(probs, self.max_rank).indices

            if token_id in topk:
                accepted_tokens += 1
            else:
                break
        return accepted_tokens,new_kv_cache