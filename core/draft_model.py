import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

class DraftModel:
    def __init__(self,model_name:str,device="cpu",dtype:torch.dtype=torch.float16,temperature:float=0.7,top_p:float =0.9,top_k:int=20):
        self.tokenizer =AutoTokenizer.from_pretrained(model_name)
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
    @torch.no_grad()    
    def init_kv_cache(self,input_ids):
        cache = DynamicCache()
        input_ids = input_ids.to(self.device)
        
        outputs = self.model(input_ids=input_ids,past_key_values=cache, use_cache=True,return_dict=True)
        
        self.kv_cache = outputs.past_key_values
        self.position = input_ids.shape[1]
        return outputs.logits[:,-1,:]
        
    @torch.no_grad()
    def forward_next(self,input_ids):
        if input_ids.shape[-1] != 1:
            raise RuntimeError(
            f"forward_next expects exactly 1 token, got {input_ids.shape}"
        )
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids, past_key_values=self.kv_cache, use_cache=True,return_dict=True)
        self.kv_cache = outputs.past_key_values
        self.position += 1
        return outputs.logits[:,-1,:]
        
    
    def sample_token(self, logits):
        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = torch.softmax(logits, dim=-1)

        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            sorted_probs[sorted_indices_to_remove] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            return torch.multinomial(probs, num_samples=1)

        if self.top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
            return top_k_indices.gather(-1, sampled_idx)

        return torch.multinomial(probs, num_samples=1)


    def rollback_kv_cache(self, prefix_length):
        self.kv_cache.crop(prefix_length)
        self.position = prefix_length
        
    def reset_kv_cache(self):
        self.kv_cache = None
        self.position = 0

    
    
        