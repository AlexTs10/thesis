import torch.nn as nn
from model_parts import Encoder, Transformer, Avg_Pool_Decoder
from utils import preprocess

class motionGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.encoder = Encoder(config=self.config)
        self.transformer = Transformer(config=self.config)
        #self.lm_head = nn.Linear(config.n_emb, config.output_size, bias=False)
        #self.lm_head_act = nn.GELU()
        self.decoder = Avg_Pool_Decoder(config=self.config)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, data_batch):
        
        x = self.encoder(data_batch)
        x = self.transformer(x)
        #x = self.lm_head_act(x)
        x = self.decoder(x)        
        return x 

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
