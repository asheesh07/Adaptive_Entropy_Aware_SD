class acceptance_criterion:
    def accept(self,target_token,draft_token):
        if hasattr(target_token,'item'):
            target_token = target_token.item()
        if hasattr(draft_token,'item'):
            draft_token = draft_token.item()
        return target_token == draft_token