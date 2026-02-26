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
        acceptance_init=1.0,
    ):
        self.draft_model = draft_model
        self.target_model = target_model

        self.entropy_calculator = EntropyCalculator()
        self.acceptance_tracker = AcceptanceTracker(acceptance_alpha, acceptance_init)

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
        self.verifier = ParallelVerifier(self.target_model,self.draft_model,self.rejection_sampler)
        
        self.performance_tracker = PerformanceTracker()
        self.quality_evaluator = QualityEvaluator()

        self.k_history = []
        self.acceptance_log = []

    # speculative_engine.py — decode() only

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, max_tokens: int):
        self.performance_tracker.reset()
        self.quality_evaluator.reset()
        self.performance_tracker.start()

        self.draft_model.reset_kv_cache()
        self.target_model.kv_cache = None
        self.target_model.position = 0

        draft_logits = self.draft_model.init_kv_cache(input_ids)
        self.target_model.init_kv_cache(input_ids)

        output_ids = input_ids.clone()

        for _ in range(max_tokens):

            z = self.entropy_calculator.compute(draft_logits)
            k = self.k_controller.decide_k(
                entropy=z,
                acceptance_rate=self.acceptance_tracker.value,
            )
            self.k_history.append(k)

            if k == 0:
                logits = self.target_model.forward_next(output_ids[:, -1:])
                next_token = self.target_model.select_tokens(logits)
                output_ids = torch.cat([output_ids, next_token], dim=1)
                draft_logits = self.draft_model.forward_next(
                    next_token.to(self.draft_model.device)
                )
                self.performance_tracker.record_target_forward(1)
                self.performance_tracker.record_tokens(1)
                continue

            draft_tokens = self.draft_generator.generate(k, output_ids[:, -1:])
            self.performance_tracker.record_draft_forward(k)

            n_accepted, next_token = self.verifier.verify(
                input_ids=output_ids,
                draft_tokens=draft_tokens,
            )

            self.performance_tracker.record_target_forward(1)
            self.acceptance_log.append((n_accepted, k))
            self.quality_evaluator.record_step(k, n_accepted)
            self.acceptance_tracker.update(n_accepted, k)

            if n_accepted > 0:
                output_ids = torch.cat(
                    [output_ids, draft_tokens[:, :n_accepted], next_token], dim=1
                )
            else:
                output_ids = torch.cat([output_ids, next_token], dim=1)

            self.performance_tracker.record_tokens(n_accepted + 1)
            self.threshold_adjuster.update(self.acceptance_tracker.value)

            draft_logits = self.draft_model.forward_next(
                next_token.to(self.draft_model.device)
            )

            if next_token.item() == self.draft_model.tokenizer.eos_token_id:
                break

        self.performance_tracker.stop()
        return output_ids
