import torch 
from torch.utils.data import DataLoader
import os
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoAgentDatasetVectorized
from config import GPTConfig
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from tqdm import tqdm  # Import tqdm


# define local data manager
config = GPTConfig()
os.environ["L5KIT_DATA_FOLDER"] = config.dataset_path #"/home/alexay/lyft-attn/DATASET_DIR/"
dm = LocalDataManager(None)
# load the experiment config
cfg = load_config_data("./config.yaml")
vectorizer = build_vectorizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoAgentDatasetVectorized(cfg, train_zarr, vectorizer)
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True)


folder = '/workspace/precomp_data'
os.makedirs(folder, exist_ok=True)
counter = 0
total = 80000
# Wrap your DataLoader with tqdm for a progress bar
for data in tqdm(iter(train_dataloader), total=total):
    torch.save(data, os.path.join(folder, f'{counter}.pt'))
    counter += 1
    if counter == total:
        break




