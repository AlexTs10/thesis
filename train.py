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


config = GPTConfig()
os.environ["L5KIT_DATA_FOLDER"] = confisg.dataset_path #"/home/alexay/lyft-attn/DATASET_DIR/"
# load the experiment config
cfg = load_config_data("./config.yaml")


model = MotionTransformer(config=config, cfg=cfg)
if config.device == 'cuda':
    model = torch.compile(model)


logger = TensorBoardLogger(save_dir="")
trainer = L.Trainer(logger=logger,                      
                     max_epochs=config.max_epochs,  
                    # precision="16-mixed",                     
                     accelerator="auto",                    
                     devices=config.device,                      
                     strategy="auto",
                     fast_dev_run=True, # debug
                     limit_train_batches=0.1, # 10% data - debug
                     limit_val_batches=0.1, # 10% data - debug
                     num_sanity_val_steps=2, # 2 val steps debug

                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True),
                                ModelSummary(max_depth=-1)])
print('---')
trainer.fit(model=model)
print('train complete')

