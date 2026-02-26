# parallel_verifier.py — optimized incremental version

import torch
import torch.nn.functional as F


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

        # --------------------------------------------------
        # No speculation case
        # --------------------------------------------------
        if k == 0:
            logits = self.target_model.forward_next(input_ids[:, -1:])
            next_token = self.rejection_sampler.handle_bonus(
                logits, self.temperature
            )
            return 0, next_token

        # --------------------------------------------------
        # 1️⃣ Forward ONLY draft tokens using cached prefix
        # --------------------------------------------------
        target_outputs = self.target_model.model(
            input_ids=draft_tokens.to(self.target_model.device),
            past_key_values=self.target_model.kv_cache,
            use_cache=True,
            return_dict=True,
        )

        target_logits = target_outputs.logits  # (1, k, vocab)
        updated_target_cache = target_outputs.past_key_values

        # Draft logits (incremental, cached)
        draft_outputs = self.draft_model.model(
            input_ids=draft_tokens.to(self.draft_model.device),
            past_key_values=self.draft_model.kv_cache,
            use_cache=True,
            return_dict=True,
        )

        draft_logits = draft_outputs.logits
        updated_draft_cache = draft_outputs.past_key_values

        # --------------------------------------------------
        # 2️⃣ Acceptance Loop (no full recompute)
        # --------------------------------------------------
        n_accepted = 0
        next_token = None

        for i in range(k):

            t_logits_i = target_logits[:, i, :]
            d_logits_i = draft_logits[:, i, :]

            t_probs = F.softmax(t_logits_i / max(self.temperature, 1e-5), dim=-1)
            d_probs = F.softmax(d_logits_i / max(self.temperature, 1e-5), dim=-1)

            draft_token_id = draft_tokens[0, i]

            p_target = t_probs[0, draft_token_id]
            p_draft  = d_probs[0, draft_token_id]

            acceptance_prob = torch.minimum(
                torch.tensor(1.0, device=p_target.device),
                p_target / (p_draft + 1e-8),
            )

            if torch.rand(1, device=p_target.device) < acceptance_prob:
                n_accepted += 1
            else:
                next_token = self.rejection_sampler.handle(
                    target_logits=t_logits_i,
                    draft_logits=d_logits_i,
                    temperature=self.temperature,
                )
                break

        # --------------------------------------------------
        # 3️⃣ If all accepted → sample bonus token
        # --------------------------------------------------
        if next_token is None:
            bonus_logits = target_logits[:, k - 1, :]
            next_token = self.rejection_sampler.handle_bonus(
                target_logits=bonus_logits,
                temperature=self.temperature,
            )

        # --------------------------------------------------
        # 4️⃣ Update KV cache WITHOUT recompute
        # --------------------------------------------------

        # Rollback to prefix + accepted tokens
        new_length = seq_len + n_accepted

        self.target_model.kv_cache = updated_target_cache
        self.target_model.position = seq_len + k
        self.target_model.rollback_kv_cache(new_length)

        self.draft_model.kv_cache = updated_draft_cache
        self.draft_model.position = seq_len + k
        self.draft_model.rollback_kv_cache(new_length)

        # --------------------------------------------------
        # 5️⃣ Append next_token incrementally
        # --------------------------------------------------

        # Forward the sampled next token to extend cache properly
        self.target_model.forward_next(next_token)
        self.draft_model.forward_next(next_token.to(self.draft_model.device))

        return n_accepted, next_token