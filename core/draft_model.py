import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


class DraftModel:
    def __init__(
        self,
        tokenizer,
        model_name: str,
        device="cuda",
        dtype: torch.dtype = torch.float16,
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

        # return last-token logits
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
    # GREEDY DRAFT SAMPLING (FASTEST OPTION)
    # --------------------------------------------------
    def sample_token(self, logits: torch.Tensor):
        return torch.argmax(logits, dim=-1, keepdim=True)

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