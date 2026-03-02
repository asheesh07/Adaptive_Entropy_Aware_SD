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
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.kv_cache = None
        self.position = 0

    # --------------------------------------------------
    # Initialize KV cache
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
    # Incremental forward
    # --------------------------------------------------
    @torch.no_grad()
    def forward_next(self, input_token: torch.Tensor):
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
    # Stable Distribution Builder
    # --------------------------------------------------
    def build_distribution(self, logits: torch.Tensor):
        """
        Returns:
            probs: sampling distribution
            entropy: entropy computed AFTER temperature scaling
        """

        # Logit stabilization (important for fp16)
        logits = logits - logits.max(dim=-1, keepdim=True).values

        # Temperature scaling
        logits = logits / max(self.temperature, 1e-6)

        probs = F.softmax(logits, dim=-1)

        # ----- Top-p (nucleus) -----
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff = cumulative_probs > self.top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False

            sorted_probs[cutoff] = 0.0

            probs = torch.zeros_like(probs).scatter_(
                -1, sorted_indices, sorted_probs
            )

            probs = probs / probs.sum(dim=-1, keepdim=True)

        # ----- Top-k -----
        if self.top_k > 0:
            topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(-1, topk_indices, topk_probs)
            probs = new_probs

        # Entropy computed on final distribution
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        # Optional normalized entropy (recommended for adaptive SD)
        vocab_size = probs.shape[-1]
        entropy_norm = entropy / torch.log(
            torch.tensor(vocab_size, device=probs.device)
        )

        return probs, entropy, entropy_norm

    # --------------------------------------------------
    # Sample token
    # --------------------------------------------------
    def sample_from_probs(self, probs: torch.Tensor):
        return torch.multinomial(probs, num_samples=1)

    # --------------------------------------------------
    # Rollback KV cache safely
    # --------------------------------------------------
    def rollback_kv_cache(self, prefix_length: int):
        if self.kv_cache is None:
            return
        self.kv_cache.crop(prefix_length)
        self.position = prefix_length

    # --------------------------------------------------
    # Reset cache
    # --------------------------------------------------
    def reset_kv_cache(self):
        self.kv_cache = None
        self.position = 0