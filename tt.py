import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Model Definition
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        return {"val_loss": val_loss}


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_dataloader(self):
        # Dummy training data
        train_data = torch.from_numpy(np.random.randn(100, 10)).float()
        train_targets = torch.from_numpy(np.random.randint(0, 2, 100)).long()
        train_dataset = TensorDataset(train_data, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=32)
        return train_loader

    def val_dataloader(self):
        # Dummy validation data
        val_data = torch.from_numpy(np.random.randn(20, 10)).float()
        val_targets = torch.from_numpy(np.random.randint(0, 2, 20)).long()
        val_dataset = TensorDataset(val_data, val_targets)
        val_loader = DataLoader(val_dataset, batch_size=32)
        return val_loader

# Training Script
def main():
    model = MyModel()
    trainer = pl.Trainer(max_epochs=10,
                         profiler="simple")
    trainer.fit(model)

if __name__ == "__main__":
    main()
