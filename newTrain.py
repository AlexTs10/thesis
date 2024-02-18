import os
from l5kit.configs import load_config_data
import torch
from torch.utils.data import DataLoader
from config import GPTConfig
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from data import PrecompDataset, PreCompDataModule
from new_models import NewTFModel

config = GPTConfig()
os.environ["L5KIT_DATA_FOLDER"] = config.dataset_path #"/home/alexay/lyft-attn/DATASET_DIR/"
# load the experiment config
cfg = load_config_data("./config.yaml")


model = NewTFModel()
if config.device == 'cuda':
    model = torch.compile(model)

data_module = PreCompDataModule(batch_size=16)
#d = PrecompDataset("/workspace/precomp_data")
#train_d = DataLoader(d, batch_size=32, num_workers=16, pin_memory=True)

logger = TensorBoardLogger(save_dir="")
trainer = L.Trainer(logger=logger,                      
                     max_epochs=config.max_epochs,  
                     accelerator="auto",                    
                     devices="auto",                      
                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True),]
                   )

trainer.fit(model=model,
            datamodule=data_module)


