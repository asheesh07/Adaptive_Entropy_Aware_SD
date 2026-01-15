from typing import Tuple, List
import torch

KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
class CacheManager:
    def slice_kv_cache(self, kv_cache: KVCache, prefix_length) -> KVCache:
        new_cache = []
        for k, v in kv_cache:
            new_k = k[:, :, :, :prefix_length, :].contiguous()
            new_v = v[:, :, :, :prefix_length, :].contiguous()
            new_cache.append((new_k, new_v))
        return tuple(new_cache)
    @staticmethod
    def commit_prefix(self,temp_kv_cache: KVCache, accepted_tokens) -> KVCache:
        if accepted_tokens<=0:
            raise ValueError("accepted_tokens must be positive to commit prefix.")
        return CacheManager.slice_kv_cache(self,temp_kv_cache, accepted_tokens)
    @staticmethod
    def rollback_kv_cache(self, kv_cache: KVCache, prefix_length) -> KVCache:
        return CacheManager.slice_kv_cache(self,kv_cache, prefix_length)
    
    def sync_cache(self,source_cache: KVCache):
        new_cache = []
        for k,v in source_cache:
           
            new_cache.append((k.contiguous(), v.contiguous()))
            
        return tuple(new_cache)
        
            
        