import torch

from adaptation.entropy_calculator import EntropyCalculator
from adaptation.k_controller import KController
from adaptation.acceptance import AcceptanceTracker
from adaptation.threshold_adjuster import ThresholdAdjuster

from core.draft_generation_loop import DraftGenerationLoop
from verification.parallel_verifier import ParallelVerifier
from verification.rejection_sampler import RejectionSampler
from optimization.cache_manager import CacheManager

from metrics.performance_tracker import PerformanceTracker
from metrics.quality_evaluator import QualityEvaluator


class SpeculativeEngine:
    def __init__(
        self,
        draft_model,
        target_model,
        max_k: int,
        entropy_bins,
        k_values,
        acceptance_alpha=0.1,
        acceptance_init=0.5,
    ):
        self.draft_model = draft_model
        self.target_model = target_model

        self.entropy_calculator = EntropyCalculator()
        self.acceptance_tracker = AcceptanceTracker(acceptance_alpha, acceptance_init)

        self.k_controller = KController(
            entropy_threshold=entropy_bins,
            k_values=k_values,
            k_max=max_k,
            acceptance_min=0.5,
            acceptance_max=0.8,
        )

        self.threshold_adjuster = ThresholdAdjuster(entropy_bins)

        self.draft_generator = DraftGenerationLoop(self.draft_model)
        self.verifier = ParallelVerifier(self.target_model)
        self.rejection_sampler = RejectionSampler(
            target_model=self.target_model,
            draft_model=self.draft_model,
        )

        self.performance_tracker = PerformanceTracker()
        self.quality_evaluator = QualityEvaluator()

        self.k_history = []
        self.acceptance_log = []

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, max_tokens: int):
        self.performance_tracker.reset()
        self.quality_evaluator.reset()
        self.performance_tracker.start()

        # Initialize both models with prompt
        self.draft_model.reset_kv_cache()
        draft_logits = self.draft_model.init_kv_cache(input_ids)
        self.target_model.init_kv_cache(input_ids)

        output_ids = input_ids.clone()

        for _ in range(max_tokens):

            # 1. Compute entropy from draft logits
            z = self.entropy_calculator.compute(draft_logits)

            # 2. Decide k
            k = self.k_controller.decide_k(
                entropy=z,
                acceptance_rate=self.acceptance_tracker.value,
            )
            self.k_history.append(k)

            # 3. k=0 fallback — target model only, no speculation
            if k == 0:
                logits = self.target_model.forward_next(output_ids[:, -1:])
                next_token = self.target_model.select_tokens(logits)  # not argmax
                output_ids = torch.cat([output_ids, next_token], dim=1)

                # keep draft in sync
                draft_logits = self.draft_model.forward_next(output_ids[:, -1:])

                self.performance_tracker.record_target_forward(1)
                self.performance_tracker.record_tokens(1)
                continue

            # 4. Draft generation
            draft_tokens = self.draft_generator.generate(k, output_ids[:, -1:])
            self.performance_tracker.record_draft_forward(k)

            # 5. Verify
            accepted, temp_target_kv = self.verifier.verify(draft_tokens)
            self.performance_tracker.record_target_forward(1)

            self.acceptance_log.append((accepted, k))
            self.quality_evaluator.record_step(k, accepted)

            # 6. Commit accepted tokens
            if accepted > 0:
                accepted_tokens = draft_tokens[:, :accepted]
                output_ids = torch.cat([output_ids, accepted_tokens], dim=1)
                self.target_model.kv_cache = temp_target_kv
                self.target_model.rollback_kv_cache(output_ids.shape[1])
                self.draft_model.rollback_kv_cache(output_ids.shape[1])  # sync draft
                self.performance_tracker.record_tokens(accepted)

            # 7. Handle rejection
            if accepted < k:
                next_token = self.rejection_sampler.handle(
                    last_committed_token=output_ids[:, -1:],
                )
                output_ids = torch.cat([output_ids, next_token], dim=1)
                self.performance_tracker.record_tokens(1)
                self.draft_model.rollback_kv_cache(output_ids.shape[1])  # sync draft

            # 8. Update adaptation components (once, not twice)
            self.acceptance_tracker.update(accepted, k)
            self.threshold_adjuster.update(self.acceptance_tracker.value)

            # 9. Prepare draft logits for next step
            draft_logits = self.draft_model.forward_next(output_ids[:, -1:])

        self.performance_tracker.stop()
        return output_ids