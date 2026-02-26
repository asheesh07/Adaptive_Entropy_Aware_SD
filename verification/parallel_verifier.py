import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
class ParallelVerifier:
    def __init__(self, target_model, draft_model, rejection_sampler, temperature: float = 1.0):
        self.target_model = target_model
        self.draft_model = draft_model
        self.rejection_sampler = rejection_sampler
        self.temperature = temperature

    @torch.no_grad()
    def verify(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor):
        k = draft_tokens.shape[1]
        seq_len = input_ids.shape[1]

        if k == 0:
            logits = self.target_model.forward_next(input_ids[:, -1:])
            next_token = self.rejection_sampler.handle_bonus(logits, self.temperature)
            return 0, next_token

        # ── Step 1: Run target over draft tokens ONE AT A TIME using KV cache ──
        # This is still ONE logical verification but uses cached prefix
        target_position_before = self.target_model.position
        draft_position_before  = self.draft_model.position

        target_logits_list = []
        draft_logits_list  = []

        # Get draft logits for each draft token position
        for i in range(k):
            token = draft_tokens[:, i:i+1].to(self.draft_model.device)
            d_logits = self.draft_model.forward_next(token)
            draft_logits_list.append(d_logits)

        # Rollback draft to before speculation
        self.draft_model.rollback_kv_cache(draft_position_before)

        # Get target logits for each draft token position  
        for i in range(k):
            token = draft_tokens[:, i:i+1].to(self.target_model.device)
            t_logits = self.target_model.forward_next(token)
            target_logits_list.append(t_logits)

        # Rollback target to before speculation
        self.target_model.rollback_kv_cache(target_position_before)

        # ── Step 2: Acceptance loop (no model calls) ──
        n_accepted = 0
        next_token = None

        for i in range(k):
            t_logits_i = target_logits_list[i]  # (1, vocab)
            d_logits_i = draft_logits_list[i]   # (1, vocab)

            # Align vocab sizes
            t_vocab = t_logits_i.shape[-1]
            d_vocab = d_logits_i.shape[-1]
            if d_vocab < t_vocab:
                d_logits_i = F.pad(d_logits_i, (0, t_vocab - d_vocab), value=float('-inf'))
            elif t_vocab < d_vocab:
                t_logits_i = F.pad(t_logits_i, (0, d_vocab - t_vocab), value=float('-inf'))

            t_probs = F.softmax(t_logits_i / max(self.temperature, 1e-5), dim=-1)
            d_probs = F.softmax(d_logits_i.to(t_logits_i.device) / max(self.temperature, 1e-5), dim=-1)

            draft_token_id = draft_tokens[0, i].item()
            p_target = t_probs[0, draft_token_id].item()
            p_draft  = d_probs[0, draft_token_id].item()

            acceptance_prob = min(1.0, p_target / (p_draft + 1e-8))

            if torch.rand(1).item() < acceptance_prob:
                n_accepted += 1
            else:
                next_token = self.rejection_sampler.handle(
                    target_logits=t_logits_i,
                    draft_logits=d_logits_i,
                    temperature=self.temperature,
                )
                break

        if next_token is None:
            # All accepted — get bonus token from target
            bonus_logits = self.target_model.forward_next(
                draft_tokens[:, k-1:k].to(self.target_model.device)
            )
            next_token = self.rejection_sampler.handle_bonus(bonus_logits, self.temperature)
            n_accepted_final = k
        else:
            n_accepted_final = n_accepted

        # ── Step 3: Commit accepted tokens to BOTH caches ──
        # Roll forward only accepted tokens + next_token
        self.target_model.rollback_kv_cache(target_position_before)
        self.draft_model.rollback_kv_cache(draft_position_before)

        for i in range(n_accepted_final):
            token = draft_tokens[:, i:i+1]
            self.target_model.forward_next(token.to(self.target_model.device))
            self.draft_model.forward_next(token.to(self.draft_model.device))

        # Commit next_token to both caches
        self.target_model.forward_next(next_token.to(self.target_model.device))
        self.draft_model.forward_next(next_token.to(self.draft_model.device))

        return n_accepted_final, next_token


