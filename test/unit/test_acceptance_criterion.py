from verification.acceptance_criterion import AcceptanceCriterion
import torch


def test_accept_equal_tokens():
    assert AcceptanceCriterion.accept(5, 5) is True


def test_reject_mismatch():
    assert AcceptanceCriterion.accept(5, 6) is False


def test_tensor_inputs():
    t1 = torch.tensor(7)
    t2 = torch.tensor(7)
    assert AcceptanceCriterion.accept(t1, t2) is True
