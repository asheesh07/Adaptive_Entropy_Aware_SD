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

        # ── Step 1: Run ONLY draft_tokens through both models using existing KV cache ──
        # KV cache already contains the full prompt + accepted history
        # so we only need to process the k new draft tokens — not the full sequence

        target_outputs = self.target_model.model(
            input_ids=draft_tokens.to(self.target_model.device),
            past_key_values=self.target_model.kv_cache,  # prefix already cached
            use_cache=True,
            return_dict=True,
        )
        target_logits = target_outputs.logits  # (1, k, vocab)

        draft_outputs = self.draft_model.model(
            input_ids=draft_tokens.to(self.draft_model.device),
            past_key_values=self.draft_model.kv_cache,  # prefix already cached
            use_cache=True,
            return_dict=True,
        )
        draft_logits = draft_outputs.logits  # (1, k, vocab)

        # ── Step 2: Align vocab sizes ──
        t_vocab = target_logits.shape[-1]
        d_vocab = draft_logits.shape[-1]
        if d_vocab < t_vocab:
            draft_logits = F.pad(draft_logits, (0, t_vocab - d_vocab), value=float('-inf'))
        elif t_vocab < d_vocab:
            target_logits = F.pad(target_logits, (0, d_vocab - t_vocab), value=float('-inf'))

        # ── Step 3: Acceptance loop — index i directly, not seq_len-1+i ──
        n_accepted = 0
        next_token = None

        for i in range(k):
            t_logits_i = target_logits[:, i, :]  # (1, vocab)
            d_logits_i = draft_logits[:, i, :]   # (1, vocab)

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
            bonus_logits = target_logits[:, k - 1, :]  # last position
            next_token = self.rejection_sampler.handle_bonus(
                target_logits=bonus_logits,
                temperature=self.temperature,
            )

        # ── Step 4: Rollback both caches to before draft, then commit accepted only ──
        # Rollback target and draft to position before draft tokens were fed in
        self.target_model.rollback_kv_cache(seq_len)
        self.draft_model.rollback_kv_cache(seq_len)

        # Commit only accepted tokens + next_token to both caches
        for i in range(n_accepted):
            token = draft_tokens[:, i:i+1]
            self.target_model.forward_next(token.to(self.target_model.device))
            self.draft_model.forward_next(token.to(self.draft_model.device))

        self.target_model.forward_next(next_token.to(self.target_model.device))
        self.draft_model.forward_next(next_token.to(self.draft_model.device))

        return n_accepted, next_token