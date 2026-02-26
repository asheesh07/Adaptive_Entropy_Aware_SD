import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

class TargetModel:
    def __init__(self,tokenizer,model_name:str,device:str="cpu",dtype:torch.dtype=torch.float16,temperature:float=0.8,top_p:float =0.9,top_k:int=0):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model =AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype).to(device)
        
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        self.kv_cache=None
        self.device = device
        self.dtype = dtype
        self.position = 0
        
    @torch.no_grad()
    def init_kv_cache(self,input_ids):
        input_ids = input_ids.to(self.device)
        cache=DynamicCache()
        outputs = self.model(input_ids=input_ids,past_key_values=cache, use_cache=True,return_dict=True)
        
        self.kv_cache = outputs.past_key_values
        self.position = input_ids.shape[1]
        return outputs.logits[:,-1,:]

    @torch.no_grad()
    def forward_next(self, input_ids):
        from transformers.cache_utils import Cache
        assert (
    self.kv_cache is None or isinstance(self.kv_cache, Cache)
), f"CORRUPTED KV CACHE: {type(self.kv_cache)}"


        input_ids = input_ids.to(self.device)

        assert input_ids.shape[-1] == 1, (
            f"TargetModel.forward_next expects 1 token, got {input_ids.shape}"
        )

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )

        self.kv_cache = outputs.past_key_values
        self.position += 1
        return outputs.logits[:, -1, :]

        
        
        
    
    @torch.no_grad()
    def select_tokens(self, logits):
        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = torch.softmax(logits, dim=-1)

        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
            sorted_indices_to_remove[..., 0] = False

            sorted_probs[sorted_indices_to_remove] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if self.top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
            return top_k_indices.gather(-1, sampled_idx)

        return torch.multinomial(probs, num_samples=1)

    def rollback_kv_cache(self, prefix_length):
        assert self.kv_cache is not None
        self.kv_cache.crop(prefix_length)
        self.position = prefix_length


    