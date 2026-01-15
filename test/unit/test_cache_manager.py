import torch
from optimization.cache_manager import CacheManager


def make_fake_kv(seq_len):
    k = torch.randn(1, 2, seq_len, 4)
    v = torch.randn(1, 2, seq_len, 4)
    return ((k, v),)


def test_slice_kv_cache():
    kv = make_fake_kv(seq_len=10)
    sliced = CacheManager.slice_kv_cache(kv, prefix_length=5)
    assert sliced[0][0].shape[2] == 5
    assert sliced[0][1].shape[2] == 5


def test_commit_prefix():
    kv = make_fake_kv(seq_len=8)
    committed = CacheManager.commit_prefix(kv, accepted_tokens=3)
    assert committed[0][0].shape[2] == 3
