from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoAgentDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from torch.utils.data import DataLoader,Dataset
from pathlib import Path
import numpy as np
import torch
import os 
import lightning as l

def train_dataset_load(cfg, subset_fraction = 1.0):
    dm = LocalDataManager(None)
    vectorizer = build_vectorizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    train_dataset = EgoAgentDatasetVectorized(cfg, train_zarr, vectorizer)
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
    val_dataset = EgoAgentDatasetVectorized(cfg, eval_zarr, vectorizer, agents_mask=eval_mask, eval_mode=True)
    return val_dataset



class DataModule(l.LightningDataModule):
    def __init__(self, cfg, config):
        super().__init__()
        self.config = config 
        self.cfg = cfg

    def train_dataloader(self):
        train_dataset = train_dataset_load(cfg=self.cfg)
        return DataLoader(train_dataset, 
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        
        # setup val dataset - only first one.
        #val_dataset_setup(self.cfg)
       # load val dataset 
        val_dataset = val_dataset_load(cfg=self.cfg)
        return DataLoader(val_dataset, 
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=True)
 


class PrecompDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [file for file in os.listdir(folder_path) if file.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        data = torch.load(file_path)
        return data


class PreCompDataModule(l.LightningDataModule):
    def __init__(self, data_folder='precomp_dataset', train_folder='train_data', val_folder='val_data', batch_size: int = 32):
        super().__init__()
        self.train_folder_path = os.path.join(data_folder, train_folder)
        self.val_folder_path = os.path.join(data_folder, val_folder)
        self.batch_size = batch_size

    def train_dataloader(self):
        train_dataset = PrecompDataset(self.train_folder_path)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        val_dataset = PrecompDataset(self.val_folder_path)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True)



