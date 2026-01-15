class DummyModel:
    def __init__(self):
        self.kv_cache = "cache"
        self.position = 0

    def forward_next(self, *_):
        return "logits"

    def select_token(self, *_):
        return "token"

    def reset_kv_cache(self):
        self.kv_cache = None


from verification.rejection_sampler import RejectionSampler


def test_rejection_sampler_does_not_crash():
    target = DummyModel()
    draft = DummyModel()

    sampler = RejectionSampler(target, draft)

    token = sampler.handle(
        accepted_tokens=0,
        temp_target_kv_cache=None,
    )

    assert token == "token"
