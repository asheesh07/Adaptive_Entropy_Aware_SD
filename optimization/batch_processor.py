from typing import List, Dict, Any
import torch


class BatchProcessor:

    def __init__(self, device):
        self.device = device

    def pad_sequences(self, sequences: List[torch.Tensor], pad_token_id: int):
    
        max_len = max(seq.shape[1] for seq in sequences)

        padded = []
        masks = []

        for seq in sequences:
            pad_len = max_len - seq.shape[1]
            if pad_len > 0:
                pad = torch.full(
                    (1, pad_len),
                    pad_token_id,
                    device=self.device,
                    dtype=seq.dtype,
                )
                padded_seq = torch.cat([seq, pad], dim=1)
                mask = torch.cat(
                    [
                        torch.ones_like(seq, device=self.device),
                        torch.zeros_like(pad, device=self.device),
                    ],
                    dim=1,
                )
            else:
                padded_seq = seq
                mask = torch.ones_like(seq, device=self.device)

            padded.append(padded_seq)
            masks.append(mask)

        return torch.cat(padded, dim=0), torch.cat(masks, dim=0)

    def batch_forward(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
