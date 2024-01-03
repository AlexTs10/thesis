from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoAgentDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import torch
import os 

def train_dataset_load(cfg, subset_fraction = 1.0):
    dm = LocalDataManager(None)
    vectorizer = build_vectorizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    train_dataset_full = EgoAgentDatasetVectorized(cfg, train_zarr, vectorizer)
    train_dataset = torch.utils.data.Subset(train_dataset_full, range(int(len(train_dataset_full)* subset_fraction)))

    return train_dataset



def val_dataset_load(cfg, subset_fraction = 1.0):
    dm = LocalDataManager(None)
    vectorizer = build_vectorizer(cfg, dm)
    eval_cfg = cfg["val_data_loader"]
    eval_base_path = os.path.join(os.environ["L5KIT_DATA_FOLDER"], f"{eval_cfg['key'].split('.')[0]}_chopped_100")
    eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]
    val_dataset_full = EgoAgentDatasetVectorized(cfg, eval_zarr, vectorizer, agents_mask=eval_mask, eval_mode=True)
    val_dataset = torch.utils.data.Subset(val_dataset_full, range(int(len(val_dataset_full)* subset_fraction)))

    return val_dataset
