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
    """
    Top-level orchestration engine for Adaptive Entropy-Aware
    Speculative Decoding.
    """

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
        # -------------------------------
        # Models
        # -------------------------------
        self.draft_model = draft_model
        self.target_model = target_model

        # -------------------------------
        # Control & Adaptation
        # -------------------------------
        self.entropy_calculator = EntropyCalculator()
        self.acceptance_tracker = AcceptanceTracker(acceptance_alpha, acceptance_init)

        self.k_controller = KController(
    entropy_threshold=entropy_bins,
    k_values=k_values,
    k_max=max_k,
)


        self.threshold_adjuster = ThresholdAdjuster(entropy_bins)

        # -------------------------------
        # Execution Modules
        # -------------------------------
        self.draft_generator = DraftGenerationLoop(self.draft_model)
        self.verifier = ParallelVerifier(self.target_model)
        self.rejection_sampler = RejectionSampler(
            target_model=self.target_model,
            draft_model=self.draft_model,
        )

        # -------------------------------
        # Metrics
        # -------------------------------
        self.performance_tracker = PerformanceTracker()
        self.quality_evaluator = QualityEvaluator()

        # -------------------------------
        # Debug / analysis hooks
        # -------------------------------
        self.k_history = []
        self.acceptance_log = []

    # ======================================================
    # Main decode loop
    # ======================================================

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, max_tokens: int):
        """
        Run speculative decoding.

        Args:
            input_ids: Tensor [1, T]
            max_tokens: number of tokens to generate

        Returns:
            output_ids: Tensor [1, T + max_tokens]
        """

        # -------------------------------
        # Initialization
        # -------------------------------
        self.performance_tracker.reset()
        self.quality_evaluator.reset()

        self.performance_tracker.start()

        # Initialize both models with prompt
        self.draft_model.reset_kv_cache()
        draft_logits = self.draft_model.init_kv_cache(input_ids)
        self.target_model.init_kv_cache(input_ids)

        output_ids = input_ids.clone()

        # ==================================================
        # Decoding loop
        # ==================================================
        for _ in range(max_tokens):

            # ----------------------------------------------
            # 1. Measure entropy (draft next-token logits)
            # ----------------------------------------------
            z = self.entropy_calculator.compute(draft_logits)

            # ----------------------------------------------
            # 2. Decide speculation depth k
            # ----------------------------------------------
            k = self.k_controller.decide_k(
                entropy=z,
                acceptance_rate=self.acceptance_tracker.value,
            )
            self.k_history.append(k)
            if k == 0:
                logits = self.target_model.forward_next(
                    output_ids[:, -1:]
                )
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                output_ids = torch.cat([output_ids, next_token], dim=1)

                self.performance_tracker.record_target_forward(1)
                self.performance_tracker.record_tokens(1)

                continue
            

            # ----------------------------------------------
            # 3. Draft generation (blind execution)
            # ----------------------------------------------
            draft_tokens = self.draft_generator.generate(k, output_ids[:, -1:])

            self.performance_tracker.record_draft_forward(k)

            # ----------------------------------------------
            # 4. Parallel verification (ONE target forward)
            # ----------------------------------------------
            accepted, temp_target_kv = self.verifier.verify(draft_tokens)
            self.performance_tracker.record_target_forward(1)

            self.acceptance_log.append((accepted, k))
            self.quality_evaluator.record_step(k, accepted)
            self.acceptance_tracker.update(accepted, k)

            # ----------------------------------------------
            # 5. Commit accepted prefix (if any)
            # ----------------------------------------------
            if accepted > 0:
                accepted_tokens = draft_tokens[:, :accepted]

                output_ids = torch.cat(
                    [output_ids, accepted_tokens], dim=1
                )

                self.target_model.kv_cache = temp_target_kv
                self.target_model.rollback_kv_cache(output_ids.shape[1])

                self.performance_tracker.record_tokens(accepted)

            # ----------------------------------------------
            # 6. Handle rejection (if any)
            # ----------------------------------------------
            if accepted < k:
                next_token = self.rejection_sampler.handle(
                last_committed_token=output_ids[:, -1:],
            )

                output_ids = torch.cat([output_ids, next_token], dim=1)
                self.performance_tracker.record_tokens(1)
                self.draft_model.kv_cache.crop(len(self.draft_model.kv_cache) - (k - accepted))


            # ----------------------------------------------
            # 7. Prepare next-step draft logits
            # ----------------------------------------------
            draft_logits = self.draft_model.forward_next(
                output_ids[:, -1:]
            )

        # ==================================================
        # Finish
        # ==================================================
        self.performance_tracker.stop()
        return output_ids
