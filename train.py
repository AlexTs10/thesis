import os
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoAgentDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from torch.utils.data import DataLoader
import torch

"""
os.environ["L5KIT_DATA_FOLDER"] = "/home/alexay/lyft-attn/DATASET_DIR/"
# define local data manager
dm = LocalDataManager(None)
# load the experiment config
cfg = load_config_data("config.yaml")
print("Configuration loaded.")
vectorizer = build_vectorizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoAgentDatasetVectorized(cfg, train_zarr, vectorizer)
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
"""
#data_batch = next(iter(train_dataloader))
data_batch = torch.load('data_batch.pt')
from config import GPTConfig
from models import motionGPT


config = GPTConfig()
model = motionGPT(config=config)
out = model(data_batch)