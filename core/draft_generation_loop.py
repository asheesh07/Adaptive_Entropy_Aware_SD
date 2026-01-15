
class DraftGenerationLoop:
    def __init__(self, draft_model):
        self.draft_model = draft_model
    
    def generate(self,k):
        draft_tokens =[]
        for i in range(k):
            logits = self.draft_model.forward_next(draft_tokens[-1] if draft_tokens else None)
            token =self.draft_model.sample_token(logits)
            self.draft_model.append(token)
            draft_tokens.append(token)
            
        return draft_tokens
            
            