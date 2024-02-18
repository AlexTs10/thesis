import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from loss_func import mse_loss
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_transformer_layers, 
                 input_dim: int = 2, 
                 num_agents: int = 50, 
                 num_timesteps: int = 20, 
                 num_future_timesteps: int = 50 ):
        super().__init__()
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        self.num_future_timesteps = num_future_timesteps
        self.hidden_dim = hidden_dim

        # Temporal encoding layer for initial representation of each agent's trajectory
        self.temporal_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Linear layer for integrating features to create a global context
        self.integration_layer = nn.Linear(hidden_dim, hidden_dim)

        # Decoder for predicting future trajectories
        self.decoder = nn.Linear(hidden_dim * num_timesteps * num_agents, num_agents * num_future_timesteps * 2)  # Predicting x, y for future timesteps

    def forward(self, x):
        # x shape: (batch_size, num_agents, num_timesteps, xy_points)
        batch_size = x.size(0)
        
        # Flatten batch and agents dimensions for temporal encoding
        x_flattened = x.view(batch_size * self.num_agents, self.num_timesteps, -1)
        temporal_encoded, _ = self.temporal_encoder(x_flattened)
        # Reshape to get (batch_size, num_agents * num_timesteps, hidden_dim)
        temporal_encoded = temporal_encoded.reshape(batch_size, self.num_agents * self.num_timesteps, self.hidden_dim)
        #print('temporal_encoded', temporal_encoded.shape)
        
        # Apply Transformer Encoder for spatial-temporal feature integration
        transformer_encoded = self.transformer_encoder(temporal_encoded)
        
        # Optional: Further integrate transformer-encoded features
        integrated_features = self.integration_layer(transformer_encoded)
        #print('integrated_feaures', integrated_features.shape)
        
        # Flatten the integrated features for decoding
        flattened_features = integrated_features.view(batch_size, -1)
        
        # Decode future trajectories
        future_trajectories = self.decoder(flattened_features)
        # Reshape future_trajectories to expected output format: (batch_size, num_agents, num_future_timesteps, 2)
        future_trajectories = future_trajectories.view(batch_size, self.num_agents, self.num_future_timesteps, 2)
        
        return future_trajectories
    

class NewTFModel(L.LightningModule):
    def __init__(self,
                #config, 
                #cfg
                ):
        super().__init__()
        self.save_hyperparameters()
        #self.config = config 
        #self.cfg = cfg
        self.torch_model = TransformerTrajectoryPredictor(hidden_dim=32, num_transformer_layers=1) 

    
    def training_step(self, batch, batch_idx):
        # Remove the second dimension from each tensor in the batch if its size is 1
        for key in batch.keys():
            # Check if the second dimension is 1
            if batch[key].dim() > 1 and batch[key].size(1) == 1:
                batch[key] = batch[key].squeeze(1)
    
        # Move batch to the same device as the model
        batch = {k: v.to(self.device) for k, v in batch.items()}
    
        # forward pass
        pred = self.torch_model(batch['all_other_agents_history_positions'])
        # loss calculation
        loss = mse_loss(model_output=pred,
                        ground_truth=batch['all_other_agents_future_positions'],
                        availability_mask=batch['all_other_agents_future_availability'])
    
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):

        # Move batch to the same device as the model
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # forward pass
        pred = self.torch_model(batch['all_other_agents_history_positions'])
        # loss calculation
        loss = mse_loss(model_output=pred,
                        ground_truth=batch['all_other_agents_future_positions'],
                        availability_mask=batch['all_other_agents_future_availability'])

        self.log("val_loss", loss)
        return loss
    

    
    def configure_optimizers(self):
        # Configure the AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)

        # Define the scheduler
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 'epoch' or 'step'
                'frequency': 1,
            }
        }