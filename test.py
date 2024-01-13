from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoAgentDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.configs import load_config_data
import os 
import numpy as np 
from torch.utils.data import DataLoader
from config import GPTConfig
from pl_models import MotionTransformer
import torch 
from tqdm import tqdm 
from tempfile import gettempdir
from l5kit.evaluation import write_pred_csv
from l5kit.geometry import transform_points
import yaml 
import glob 

 ## -- -- -- ##
checkpoint_path = "/home/alexay/lyft-attn/thesis/lightning_logs/version_13/checkpoints/*"
#h_param_yaml_path = "lightning_logs/version_11/hparams.yaml"
# load the experiment config
cfg = load_config_data("./config.yaml")
config = GPTConfig()
os.environ["L5KIT_DATA_FOLDER"] = config.dataset_path

# Load Test Dataset 
dm = LocalDataManager(None)
vectorizer = build_vectorizer(cfg, dm)
test_zarr_path = os.path.join(os.environ["L5KIT_DATA_FOLDER"], "scenes/test.zarr")
test_zarr = ChunkedDataset(test_zarr_path).open()
test_mask = np.load(f"{cfg['dataset_path']}/scenes/mask.npz")["arr_0"]
test_dataset = EgoAgentDatasetVectorized(cfg, test_zarr, vectorizer, agents_mask=test_mask, eval_mode=True)

test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

#with open(h_param_yaml_path, 'r') as file:
#    hyperparams = yaml.safe_load(file)
model = MotionTransformer.load_from_checkpoint(checkpoint_path=glob.glob(checkpoint_path)[0], map_location=torch.device('cpu')) #, cfg=cfg, config=config)
model.eval()
torch.set_grad_enabled(False)
# ----------------------------- #
# EVAL LOOP
# ==== EVAL LOOP

# store information for evaluation
future_coords_offsets_pd = []
timestamps = []
agent_ids = []
confidences = []

progress_bar = tqdm(test_dataloader)
for data in progress_bar:
    preds, conf = model(data)
    
    # convert agent coordinates into world offsets
    agents_coords = preds.detach().cpu().numpy().reshape(config.batch_size, 3*50, 2)
    world_from_agents = data["world_from_agent"].cpu().numpy()
    centroids = data["centroid"].unsqueeze(1).unsqueeze(2).cpu().numpy() # (bs, 1, 1, 2)
    coords_offset = transform_points(agents_coords, world_from_agents).reshape(config.batch_size, 3, 50, 2) - centroids#[:, None, :2]
    
    future_coords_offsets_pd.append(np.stack(coords_offset))
    timestamps.append(data["timestamp"].cpu().numpy().copy())
    agent_ids.append(data["track_id"].cpu().numpy().copy())
    confidences.append(conf)    

# -- -- #
pred_path = f"{gettempdir()}/pred.csv"

write_pred_csv(pred_path,
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd),
               confs=np.concatenate(confidences)
              )

