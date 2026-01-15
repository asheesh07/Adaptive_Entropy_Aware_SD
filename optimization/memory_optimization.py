import torch
import gc


class MemoryOptimizer:
    """
    Utilities to reduce GPU memory pressure during speculative decoding.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    # --------------------------------------------------
    # 1. Precision helpers
    # --------------------------------------------------

    @staticmethod
    def ensure_half_precision(model):
        """
        Ensure model runs in fp16/bf16 for lower memory usage.
        """
        for param in model.parameters():
            if param.dtype not in (torch.float16, torch.bfloat16):
                param.data = param.data.half()

    # --------------------------------------------------
    # 2. Explicit cleanup after rejection
    # --------------------------------------------------

    @staticmethod
    def cleanup_temporary_tensors():
        """
        Aggressively free temporary tensors after rollback.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # 3. KV cache offloading (optional, conservative)
    # --------------------------------------------------

    @staticmethod
    def offload_kv_cache_to_cpu(kv_cache):
        """
        Move KV cache to CPU to free GPU memory.
        """
        if kv_cache is None:
            return None

        cpu_cache = []
        for k, v in kv_cache:
            cpu_cache.append((k.cpu(), v.cpu()))
        return tuple(cpu_cache)

    @staticmethod
    def load_kv_cache_to_gpu(kv_cache, device):
        """
        Move KV cache back to GPU.
        """
        if kv_cache is None:
            return None

        gpu_cache = []
        for k, v in kv_cache:
            gpu_cache.append((k.to(device), v.to(device)))
        return tuple(gpu_cache)

    # --------------------------------------------------
    # 4. Memory stats (debug / benchmarks)
    # --------------------------------------------------

    @staticmethod
    def memory_stats():
        """
        Return lightweight GPU memory stats.
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_MB": torch.cuda.memory_allocated() / 1024**2,
            "reserved_MB": torch.cuda.memory_reserved() / 1024**2,
        }
