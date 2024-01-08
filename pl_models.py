from typing import Any
import lightning as L
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models import motionGPT
from utils import criterion
from data import train_dataset_load, val_dataset_load
from utils import val_dataset_setup
from torch.utils.data import DataLoader

class MotionTransformer(L.LightningModule):
    def __init__(self, config, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 
        self.cfg = cfg
        self.torch_model = motionGPT(self.config)    
    
    def training_step(self, batch, batch_idx):
        # Remove the second dimension from each tensor in the batch if its size is 1
        for key in batch.keys():
            # Check if the second dimension is 1
            if batch[key].dim() > 1 and batch[key].size(1) == 1:
                batch[key] = batch[key].squeeze(1)
    
        # Move batch to the same device as the model
        batch = {k: v.to(self.device) for k, v in batch.items()}
    
        # forward pass
        pred, conf = self.torch_model(batch)
        # loss calculation
        loss = criterion(gt=batch['target_positions'], 
                         pred=pred, 
                         confidences=conf, 
                         avails=batch['target_availabilities'])
    
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):

        # Move batch to the same device as the model
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # forward pass
        pred, conf = self.torch_model(batch)
        # loss calculation
        loss = criterion(gt=batch['target_positions'], 
                         pred=pred, 
                         confidences=conf, 
                         avails=batch['target_availabilities'])

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
    
