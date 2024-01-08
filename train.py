import os
from l5kit.configs import load_config_data
import torch
import torch.nn as nn
from config import GPTConfig
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary
from pl_models import MotionTransformer
from data import DataModule
from data import PrecompDataset
from torch.utils.data import DataLoader, Dataset

config = GPTConfig()
os.environ["L5KIT_DATA_FOLDER"] = config.dataset_path #"/home/alexay/lyft-attn/DATASET_DIR/"
# load the experiment config
cfg = load_config_data("./config.yaml")


model = MotionTransformer(config=config, cfg=cfg)
if config.device == 'cuda':
    model = torch.compile(model)

#data_module = DataModule(config=config, cfg=cfg)
d = PrecompDataset("/workspace/precomp_data")
train_d = DataLoader(d, batch_size=32, num_workers=16, pin_memory=True)

logger = TensorBoardLogger(save_dir="")
trainer = L.Trainer(logger=logger,                      
                     max_epochs=config.max_epochs,  
                    # precision="16-mixed",                     
                     accelerator="auto",                    
                     devices="auto",                      
                     #strategy="auto",
                     #fast_dev_run=True, # debug
                     #limit_train_batches=0.1, # 10% data - debug
                     #limit_val_batches=0.1, # 10% data - debug
                     #num_sanity_val_steps=2, # 2 val steps debug

                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True),
                                ModelSummary(max_depth=-1)])
print('---')
trainer.fit(model=model, 
            #datamodule=data_module,
            train_dataloaders = train_d
           )
print('train complete')

