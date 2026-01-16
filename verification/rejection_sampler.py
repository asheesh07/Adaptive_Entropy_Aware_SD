import torch
class RejectionSampler:
    def __init__(self,target_model,draft_model):
        self.target_model=target_model
        self.draft_model=draft_model
        
    def handle(self,last_committed_token: torch.Tensor,):
        

        logits = self.target_model.forward_next(last_committed_token)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token