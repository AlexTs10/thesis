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
    
    def train_dataloader(self):
        train_dataset = train_dataset_load(cfg=self.cfg)
        return DataLoader(train_dataset, batch_size=self.config.batch_size)

    def val_dataloader(self):
        
        # setup val dataset - only first one.
        val_dataset_setup(self.cfg)
       # load val dataset 
        val_dataset = val_dataset_load(cfg=self.cfg)
        return DataLoader(val_dataset, batch_size=self.config.batch_size)
 

