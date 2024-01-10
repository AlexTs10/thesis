from dataclasses import dataclass, field
import torch

#d = 'cuda' if torch.cuda.is_available() else 'cpu'
@dataclass
class GPTConfig:
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device: str = field(default=None, init=False)


    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    
    def __post_init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    n_layer: int = 12 #3
    n_head: int = 12 
    n_emb: int = 768 # 24 
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    dataset_path: str = "/workspace/DATASET_EDITED"
    n_coord = 3
    output_size: int = 303 ## ???
    n_obj: int = 161 # agents + map
    n_t: int = 20

    batch_size: int = 32
    num_workers: int = 16
    max_epochs: int = 25
    
#config = GPTConfig()
