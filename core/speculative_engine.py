import torch

class SpeculativeEngine:
    def __init__(
        self,
        draft_model,
        target_model,
        max_k: int,
        entropy_bins,
        k_values,
        acceptance_alpha=0.1,
        acceptance_init=1.0,
    ):
        self.draft_model = draft_model
        self.target_model = target_model

        from adaptation.acceptance import AcceptanceTracker
        from adaptation.k_controller import KController
        from adaptation.threshold_adjuster import ThresholdAdjuster
        from core.draft_generation_loop import DraftGenerationLoop
        from verification.parallel_verifier import ParallelVerifier
        from verification.rejection_sampler import RejectionSampler
        from metrics.performance_tracker import PerformanceTracker
        from metrics.quality_evaluator import QualityEvaluator

        self.acceptance_tracker = AcceptanceTracker(
            acceptance_alpha, acceptance_init
        )

        self.k_controller = KController(
            entropy_threshold=entropy_bins,
            k_values=k_values,
            k_max=max_k,
        )

        self.threshold_adjuster = ThresholdAdjuster(entropy_bins)

        self.draft_generator = DraftGenerationLoop(self.draft_model)

        self.rejection_sampler = RejectionSampler(
            target_model=self.target_model
        )

        self.verifier = ParallelVerifier(
            self.target_model,
            self.draft_model,
            self.rejection_sampler
        )

        self.performance_tracker = PerformanceTracker()
        self.quality_evaluator = QualityEvaluator()

        self.k_history = []
        self.acceptance_log = []

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------
    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, max_tokens: int):

        self.performance_tracker.reset()
        self.quality_evaluator.reset()
        self.performance_tracker.start()

        # Reset caches
        self.draft_model.reset_kv_cache()
        self.target_model.reset_kv_cache()

        # Prime both models
        draft_logits = self.draft_model.init_kv_cache(input_ids)
        self.target_model.init_kv_cache(input_ids)

        output_ids = input_ids.clone()

        for _ in range(max_tokens):

            # ------------------------------------------
            # 1️⃣ Compute entropy from draft distribution
            # ------------------------------------------
            draft_probs, entropy, entropy_norm = \
                self.draft_model.build_distribution(draft_logits)

            z = entropy_norm

            k = self.k_controller.decide_k(
                entropy=z,
                acceptance_rate=self.acceptance_tracker.value,
            )

            self.k_history.append(k)

            # ------------------------------------------
            # 2️⃣ No drafting case (fallback)
            # ------------------------------------------
            if k == 0:

                target_logits = self.target_model.forward_next(
                    output_ids[:, -1:]
                )

                target_probs, _, _ = \
                    self.target_model.build_distribution(target_logits)

                next_token = self.target_model.sample_from_probs(
                    target_probs
                )

                output_ids = torch.cat(
                    [output_ids, next_token], dim=1
                )

                draft_logits = self.draft_model.forward_next(
                    next_token.to(self.draft_model.device)
                )

                self.performance_tracker.record_target_forward(1)
                self.performance_tracker.record_tokens(1)

                continue

            # ------------------------------------------
            # 3️⃣ Draft generation
            # ------------------------------------------
            draft_tokens = self.draft_generator.generate(
                k, output_ids[:, -1:]
            )

            self.performance_tracker.record_draft_forward(k)

            # ------------------------------------------
            # 4️⃣ Verification
            # ------------------------------------------
            n_accepted, next_token = self.verifier.verify(
                input_ids=output_ids,
                draft_tokens=draft_tokens,
            )

            self.performance_tracker.record_target_forward(1)

            self.acceptance_log.append((n_accepted, k))
            self.quality_evaluator.record_step(k, n_accepted)

            # Update acceptance rate properly
            self.acceptance_tracker.update(n_accepted + 1, k + 1)

            # ------------------------------------------
            # 5️⃣ Append accepted tokens
            # ------------------------------------------
            if n_accepted > 0:
                output_ids = torch.cat(
                    [
                        output_ids,
                        draft_tokens[:, :n_accepted],
                        next_token,
                    ],
                    dim=1,
                )
            else:
                output_ids = torch.cat(
                    [output_ids, next_token], dim=1
                )

            # ------------------------------------------
            # 6️⃣ Rollback rejected draft tokens
            # ------------------------------------------
            if n_accepted < k:
                rollback_to = (
                    self.draft_model.position - (k - n_accepted)
                )
                self.draft_model.rollback_kv_cache(
                    rollback_to
                )

            # ------------------------------------------
            # 7️⃣ Advance draft cache with target token
            # ------------------------------------------
            draft_logits = self.draft_model.forward_next(
                next_token.to(self.draft_model.device)
            )

            self.performance_tracker.record_tokens(
                n_accepted + 1
            )

            self.threshold_adjuster.update(
                self.acceptance_tracker.value
            )

            if next_token.item() == \
                self.draft_model.tokenizer.eos_token_id:
                break

        self.performance_tracker.stop()
        return output_ids