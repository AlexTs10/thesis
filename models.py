import torch
import torch.nn as nn
from torch.nn import functional as F
import math 

from model_parts import LayerNorm, CausalSelfAttention, MLP, Block 
from utils import preprocess

class motionGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        #assert config.vocab_size is not None
        #assert config.block_size is not None
        self.config = config

        self.input_embed = nn.Linear(config.n_coord, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            #wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.n_obj * config.n_t, config.n_embd), # position embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.output_size, bias=False)
        self.lm_head_act = nn.GELU()
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, data_batch): #idx): #, targets=None):
        
        all_polys, invalid_polys = preprocess(data_batch=data_batch) # bs, num_obj*num_time, num_coord
        #b, t = idx.size()
        bs, obj_time_flattened, num_coord = all_polys.shape 
        
        all_polys_emb = self.input_embed(all_polys) 
        
        pos = torch.arange(0, obj_time_flattened, dtype=torch.long, device=self.config.device) # shape (t)

        # forward the GPT model itself
        #tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        #x = self.transformer.drop(tok_emb + pos_emb)
        x = self.transformer.drop(all_polys_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        ## ADD A FINAL ACTIVATION  
        out = self.lm_head_act(x)
        return out #logits, loss


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params