from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 12
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    n_coord = 3
    output_size: int = 150 ## ???
    n_obj: int = 161 # agents + map
    n_t: int = 20


#config = GPTConfig()
