from adaptation.k_controller import KController


def test_entropy_to_k_mapping():
    controller = KController(
        entropy_bins=[1.0, 2.0],
        k_values=[8, 4, 0],
        k_max=8,
    )

    assert controller.decide_k(entropy=0.5, acceptance_rate=1.0) == 8
    assert controller.decide_k(entropy=1.5, acceptance_rate=1.0) == 4
    assert controller.decide_k(entropy=3.0, acceptance_rate=1.0) == 0


def test_acceptance_dampening():
    controller = KController(
        entropy_bins=[1.0],
        k_values=[6, 2],
        k_max=6,
        min_acceptance=0.4,
    )

    k = controller.decide_k(entropy=0.2, acceptance_rate=0.2)
    assert k < 6
