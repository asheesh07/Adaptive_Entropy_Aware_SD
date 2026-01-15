import torch
from adaptation.entropy_calculator import EntropyCalculator


def test_entropy_low_for_peaked_distribution():
    logits = torch.tensor([[10.0, -10.0]])
    entropy = EntropyCalculator().compute(logits)
    assert entropy.item() < 0.1


def test_entropy_high_for_uniform_distribution():
    logits = torch.zeros(1, 1000)
    entropy = EntropyCalculator().compute(logits)
    assert entropy.item() > 6.0


def test_entropy_no_nan():
    logits = torch.tensor([[float("-inf"), 0.0]])
    entropy = EntropyCalculator().compute(logits)
    assert torch.isfinite(entropy).all()
