import torch
import gc


class MemoryOptimizer:
    
    def __init__(self, device: str = "cuda"):
        self.device = device


    @staticmethod
    def ensure_half_precision(model):
        
        for param in model.parameters():
            if param.dtype not in (torch.float16, torch.bfloat16):
                param.data = param.data.half()


    @staticmethod
    def cleanup_temporary_tensors():
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    @staticmethod
    def offload_kv_cache_to_cpu(kv_cache):
        
        if kv_cache is None:
            return None

        cpu_cache = []
        for k, v in kv_cache:
            cpu_cache.append((k.cpu(), v.cpu()))
        return tuple(cpu_cache)

    @staticmethod
    def load_kv_cache_to_gpu(kv_cache, device):
        
        if kv_cache is None:
            return None

        gpu_cache = []
        for k, v in kv_cache:
            gpu_cache.append((k.to(device), v.to(device)))
        return tuple(gpu_cache)

    @staticmethod
    def memory_stats():
        
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_MB": torch.cuda.memory_allocated() / 1024**2,
            "reserved_MB": torch.cuda.memory_reserved() / 1024**2,
        }
