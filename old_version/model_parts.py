import torch
import torch.nn as nn
from torch.nn import functional as F
import math 
from utils import preprocess

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_emb, config.n_emb, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.n_obj*config.n_t, config.n_obj*config.n_t))
                                        .view(1, 1, config.n_obj*config.n_t, config.n_obj*config.n_t))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_emb)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_emb, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_emb, 4 * config.n_emb, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_emb, config.n_emb, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_emb, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_emb, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

# Embedding Input Data + Position Embs
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_layer = nn.Linear(self.config.n_obj * self.config.n_coord, self.config.n_emb)
        self.pos_embedding = nn.Embedding(config.n_t, config.n_emb) # position embedding


    def forward(self, data_batch):
        all_polys, invalid_polys = preprocess(data_batch=data_batch, device=self.config.device) # (bs, T, M, 3) , (bs, M)
        #print(all_polys.shape)
        Bs, T, N_obj, _ = all_polys.size()
        
        # Flatten the last two dimensions
        x = all_polys.reshape(Bs, T, -1)  # shape becomes (Bs, T, M*3)
        #print(x.shape)
        # Apply the embedding layer to each time step
        x = self.embedding_layer(x)  # shape becomes (Bs, T, Emb_dim)

        pos = torch.arange(0, self.config.n_t, dtype=torch.long, device=self.config.device) # vector T
        pos_emb = self.pos_embedding(pos) # position embeddings of shape (T, n_emb)
        #print(pos_emb.shape)
        return (x + pos_emb)
    


class SingleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # This layer will embed each (M, 3) into an embedding space
        self.individual_embedding_layer = nn.Linear(3, self.config.n_emb)
        # Note: Depending on your specific requirements, you might want to adjust the output size of this layer
        self.embedding_layer = nn.Linear(self.config.n_obj * self.config.n_emb, self.config.n_emb)
        self.pos_embedding = nn.Embedding(config.n_t, config.n_emb) # position embedding

    def forward(self, data_batch):
        all_polys, invalid_polys = preprocess(data_batch=data_batch, device=self.config.device) # (bs, T, M, 3)
        Bs, T, N_obj, _ = all_polys.size()
        
        # New approach: Embed each (M, 3) separately for each T, then concatenate
        embedded_polys = []
        for t in range(T):
            # Embed each (M, 3) at time t
            embedded_t = self.individual_embedding_layer(all_polys[:, t, :, :])  # shape becomes (Bs, M, Emb_dim)
            embedded_polys.append(embedded_t.unsqueeze(1))

        # Concatenate embeddings for all time steps, shape becomes (Bs, T, M*Emb_dim)
        x = torch.cat(embedded_polys, dim=1)  # Note: Adjust dimension if necessary
        
        # Flatten the last two dimensions
        x = x.reshape(Bs, T, -1)  # shape becomes (Bs, T, M*Emb_dim)
        
        # Apply the embedding layer to the concatenated embeddings
        x = self.embedding_layer(x)  # shape becomes (Bs, T, Emb_dim)

        pos = torch.arange(0, self.config.n_t, dtype=torch.long, device=self.config.device) # vector T
        pos_emb = self.pos_embedding(pos) # position embeddings of shape (T, n_emb)
        
        return (x + pos_emb)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_emb, bias=config.bias),
        ))

    def forward(self, x):
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x 

class Avg_Pool_Decoder(nn.Module):
    def __init__(self, 
                 config, 
                 num_modes: int = 3, 
                 future_steps: int = 50):
        super().__init__()

        self.config = config
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.input_dim = config.n_emb
        self.output_dim = config.output_size
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        # Assuming x is of shape (bs, T, Emb)
        bs, T, Emb = x.shape
        # Apply average pooling across the sequence dimension
        x_pooled = torch.mean(x, dim=1)  # Shape becomes (bs, Emb)

        # Apply a linear transformation
        x_transformed = self.linear(x_pooled)  # Shape becomes (bs, Out_dim) - (bs, 303)

        #bs, future_len, num_coords = gt.shape
        # Reshape pred to (bs, num_modes, future_len, num_coords)
        coord_pred = x_transformed[:, :self.num_modes * self.future_steps * 2].reshape(bs, self.num_modes, self.future_steps, 2)
        #print(coord_pred.shape)
        # Extract confidences from the last 3 values in pred
        confidences = x_transformed[:, -self.num_modes:].reshape(bs, self.num_modes) # Bs, 3
        # Applying softmax for each batch
        conf = F.softmax(confidences, dim=1)
        #print(conf.shape)
        return coord_pred, conf
        

        