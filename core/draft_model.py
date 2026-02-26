import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


class DraftModel:
    def __init__(
        self,
        tokenizer,
        model_name: str,
        device="cuda",
        dtype: torch.dtype = torch.float16,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
    ):
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device
        self.dtype = dtype

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.kv_cache = None
        self.position = 0

    # --------------------------------------------------
    # Initialize KV cache with full prefix
    # --------------------------------------------------
    @torch.no_grad()
    def init_kv_cache(self, input_ids: torch.Tensor):
        input_ids = input_ids.to(self.device)

        cache = DynamicCache()

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        self.kv_cache = outputs.past_key_values
        self.position = input_ids.shape[1]

        return outputs.logits[:, -1, :]

    # --------------------------------------------------
    # Incremental forward (1 token at a time)
    # --------------------------------------------------
    @torch.no_grad()
    def forward_next(self, input_token: torch.Tensor):
        if input_token.shape[-1] != 1:
            raise RuntimeError(
                f"forward_next expects exactly 1 token, got {input_token.shape}"
            )

        input_token = input_token.to(self.device)

        outputs = self.model(
            input_ids=input_token,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )

        self.kv_cache = outputs.past_key_values
        self.position += 1

        return outputs.logits[:, -1, :]

    # --------------------------------------------------
    # PROPER SAMPLING (MATCH TARGET SETTINGS)
    # --------------------------------------------------
    def sample_token(self, logits: torch.Tensor):

        logits = logits / max(self.temperature, 1e-5)

        probs = F.softmax(logits, dim=-1)

        # Top-p (nucleus) sampling
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove = sorted_indices_to_remove.clone()
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            sorted_probs[sorted_indices_to_remove] = 0.0

            probs = torch.zeros_like(probs).scatter_(
                -1, sorted_indices, sorted_probs
            )

            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Top-k sampling
        if self.top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
            return top_k_indices.gather(-1, sampled_idx)

        return torch.multinomial(probs, num_samples=1)

    # --------------------------------------------------
    # Rollback KV cache after rejection
    # --------------------------------------------------
    def rollback_kv_cache(self, prefix_length: int):
        self.kv_cache.crop(prefix_length)
        self.position = prefix_length

    # --------------------------------------------------
    # Reset cache
    # --------------------------------------------------
    def reset_kv_cache(self):
        self.kv_cache = None
        self.position = 0