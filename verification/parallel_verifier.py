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

        # ── Step 1: Record position BEFORE we touch anything ──
        position_before = seq_len  # both models should be at seq_len after init/last commit

        # ── Step 2: Collect target logits using forward_next (fast, uses KV cache) ──
        target_logits_list = []
        for i in range(k):
            token = draft_tokens[:, i:i+1].to(self.target_model.device)
            logits = self.target_model.forward_next(token)  # (1, vocab)
            target_logits_list.append(logits)

        # Rollback target to before draft tokens
        self.target_model.rollback_kv_cache(position_before)

        # ── Step 3: Collect draft logits using forward_next (fast, uses KV cache) ──
        draft_logits_list = []
        for i in range(k):
            token = draft_tokens[:, i:i+1].to(self.draft_model.device)
            logits = self.draft_model.forward_next(token)  # (1, vocab)
            draft_logits_list.append(logits)

        # Rollback draft to before draft tokens
        self.draft_model.rollback_kv_cache(position_before)

        # ── Step 4: Acceptance loop (zero model calls) ──
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
            # All accepted — bonus token from last target logits
            next_token = self.rejection_sampler.handle_bonus(
                target_logits=target_logits_list[k - 1],
                temperature=self.temperature,
            )

        # ── Step 5: Commit accepted tokens + next_token to both caches ──
        # Both caches are at position_before thanks to rollback
        for i in range(n_accepted):
            token = draft_tokens[:, i:i+1]
            self.target_model.forward_next(token.to(self.target_model.device))
            self.draft_model.forward_next(token.to(self.draft_model.device))

        # Commit next_token
        self.target_model.forward_next(next_token.to(self.target_model.device))
        self.draft_model.forward_next(next_token.to(self.draft_model.device))

        return n_accepted, next_token
