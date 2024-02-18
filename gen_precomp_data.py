import os
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoAgentDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
import numpy as np
import torch
from tempfile import gettempdir
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
import lightning as l
from config import GPTConfig
from data import val_dataset_load, train_dataset_load

torch.manual_seed(123)
np.random.seed(123)
config = GPTConfig()
os.environ["L5KIT_DATA_FOLDER"] = config.dataset_path 

# define local data manager
dm = LocalDataManager(None)
# load the experiment config
cfg = load_config_data("../solution/config.yaml")


train_dataset = train_dataset_load(cfg=cfg)
train_dataloader = DataLoader(train_dataset, 
                          batch_size=1,
                          num_workers=2,
                          pin_memory=True,
                             shuffle=True)

val_dataset = val_dataset_load(cfg=cfg)
val_dataloader = DataLoader(val_dataset, 
                          batch_size=1,
                          num_workers=2,
                          pin_memory=True,
                            shuffle=True)

# Create folders for train and validation data
data_folder = 'precompt_dataset'
train_folder = 'train_data'
val_folder = 'val_data'
os.makedirs(os.path.join(data_folder, train_folder), exist_ok=True)
os.makedirs(os.path.join(data_folder, val_folder), exist_ok=True)

# Assuming 0.5% of 4M training data = 20K ~ 2.4GB
# And a similar approach for validation data if needed
train_counter = 0
val_counter = 0

# Save train data
for data in iter(train_dataloader):
    torch.save(data, os.path.join(data_folder, train_folder, f'{train_counter}.pt'))
    train_counter += 1
    if train_counter == 20000:  # Adjust this number based on how many samples you want to save
        break

# Save validation data
for data in iter(val_dataloader):
    torch.save(data, os.path.join(data_folder, val_folder, f'{val_counter}.pt'))
    val_counter += 1
    if val_counter == 4000:  # Adjust this number based on your validation data size
        break


