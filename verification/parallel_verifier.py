# parallel_verifier.py — adapted to unified distribution API

import torch


class ParallelVerifier:
    def __init__(self, target_model, draft_model, rejection_sampler):
        self.target_model = target_model
        self.draft_model = draft_model
        self.rejection_sampler = rejection_sampler

    @torch.no_grad()
    def verify(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor):

        k = draft_tokens.shape[1]
        seq_len = input_ids.shape[1]

        # --------------------------------------------------
        # No speculation case
        # --------------------------------------------------
        if k == 0:
            logits = self.target_model.forward_next(input_ids[:, -1:])
            probs, _, _ = self.target_model.build_distribution(logits)
            next_token = self.target_model.sample_from_probs(probs)
            return 0, next_token

        # --------------------------------------------------
        # 1️⃣ Forward draft tokens in parallel
        # --------------------------------------------------
        target_outputs = self.target_model.model(
            input_ids=draft_tokens.to(self.target_model.device),
            past_key_values=self.target_model.kv_cache,
            use_cache=True,
            return_dict=True,
        )

        target_logits = target_outputs.logits
        updated_target_cache = target_outputs.past_key_values

        draft_outputs = self.draft_model.model(
            input_ids=draft_tokens.to(self.draft_model.device),
            past_key_values=self.draft_model.kv_cache,
            use_cache=True,
            return_dict=True,
        )

        draft_logits = draft_outputs.logits
        updated_draft_cache = draft_outputs.past_key_values

        # --------------------------------------------------
        # 2️⃣ Acceptance Loop
        # --------------------------------------------------
        n_accepted = 0
        next_token = None

        for i in range(k):

            t_logits_i = target_logits[:, i, :]
            d_logits_i = draft_logits[:, i, :]

            # Use unified distribution builder
            t_probs, _, _ = self.target_model.build_distribution(t_logits_i)
            d_probs, _, _ = self.draft_model.build_distribution(d_logits_i)

            draft_token_id = draft_tokens[0, i]

            p_target = t_probs[0, draft_token_id]
            p_draft = d_probs[0, draft_token_id]

            acceptance_prob = torch.minimum(
                torch.tensor(1.0, device=p_target.device),
                p_target / (p_draft + 1e-8),
            )

            if torch.rand(1, device=p_target.device) < acceptance_prob:
                n_accepted += 1
            else:
                # rejection sampling correction
                next_token = self.rejection_sampler.handle(
                    target_logits=t_logits_i,
                    draft_logits=d_logits_i,
                )
                break

        # --------------------------------------------------
        # 3️⃣ If all accepted → sample bonus token
        # --------------------------------------------------
        if next_token is None:
            bonus_logits = target_logits[:, k - 1, :]
            bonus_probs, _, _ = self.target_model.build_distribution(
                bonus_logits
            )
            next_token = self.target_model.sample_from_probs(
                bonus_probs
            )

        # --------------------------------------------------
        # 4️⃣ Rollback to prefix + accepted
        # --------------------------------------------------
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
        self.target_model.forward_next(next_token)
        self.draft_model.forward_next(
            next_token.to(self.draft_model.device)
        )

        return n_accepted, next_token