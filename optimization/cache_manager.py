from transformers.cache_utils import Cache

class CacheManager:
    @staticmethod
    def assert_valid(cache):
        if cache is None:
            return
        assert isinstance(cache, Cache), (
            f"Invalid cache type: {type(cache)}"
        )

    @staticmethod
    def commit(cache, n_tokens):
        CacheManager.assert_valid(cache)
        if n_tokens > 0:
            cache.crop(n_tokens)
        return cache

    @staticmethod
    def reset():
        return None
