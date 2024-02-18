from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoAgentDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.configs import load_config_data
import os 
import numpy as np 
from torch.utils.data import DataLoader
from config import GPTConfig
from new_models import NewTFModel
import torch 
from tqdm import tqdm 
from tempfile import gettempdir
from l5kit.evaluation import write_pred_csv
from l5kit.geometry import transform_points, build_matrix
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
model = NewTFModel.load_from_checkpoint(checkpoint_path=glob.glob(checkpoint_path)[0], map_location=torch.device('cpu')) #, cfg=cfg, config=config)
model.eval()
torch.set_grad_enabled(False)
device = 'cpu'
# ----------------------------- #
# EVAL LOOP
# ==== EVAL LOOP

# store information for evaluation
future_coords_offsets_pd = []
future_traj_confidence = []
timestamps = []
agent_ids = []
agent_of_interest_ids = []
missing_agent_of_interest_ids = []
missing_agent_of_interest_timestamp = []

# torch.isin is available only form pytorch 1.10 - defining a simple alternative
def torch_isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

# iterate over validation dataset
#progress_bar = tqdm(eval_dataloader)
progress_bar = tqdm(test_dataloader)
for data in progress_bar:
    data = {k: v.to(device) for k, v in data.items()}
    outputs = model(data['all_other_agents_history_positions']) 

    # [batch_size, max_num_agents, num_trajectories, num_timesteps, 2]
    agent_xy = outputs.unsqueeze(2) #bs, agents, traj, timesteps, 2
    # [batch_size, max_num_agents, num_trajectories, num_timesteps, 1]
    agent_yaw = data["all_other_agents_future_yaws"].unsqueeze(2)
    # [batch_size, max_num_agents, num_trajectories]
    agent_logits = torch.ones((outputs.shape[0], 50, 1)) # to be generated from model

    # [batch_size, max_num_agents, num_trajectories, num_timesteps, 3]
    agent_pos = torch.cat((agent_xy, agent_yaw), dim=-1)

    # ego-centric agent coords must be converted to world frame first
    # [batch_size, 3, 3]
    world_from_agents = data["world_from_agent"].float()
    # [batch_size]
    world_from_agents_yaw = data["yaw"].float()
    # shape of data["all_other_agents_history_positions"]: [batch_size, num_agents, num_history_frames, 2]
    # [batch_size, num_agents, 1, 3]
    agent_t0_pos_yaw = torch.cat((data["all_other_agents_history_positions"][:, :, :1],
                                  data["all_other_agents_history_yaws"][:, :, :1]), dim=-1)
    agent_t0_avail = data["all_other_agents_history_availability"][:, :, :1]
    # [batch_size, num_agents, 1, 3]
    world_agent_t0_pos_yaw = transform_points(agent_t0_pos_yaw, world_from_agents, avail=agent_t0_avail,
                                              yaw=world_from_agents_yaw)
    world_agent_pos = transform_points(agent_pos.flatten(2,3), world_from_agents, avail=agent_t0_avail).view_as(agent_pos)

    # then can be converted to agent-relative
    world_agents_t0_pos_exp = world_agent_t0_pos_yaw[..., :2]
    world_agents_t0_yaw_exp = world_agent_t0_pos_yaw[..., 2]
    # [batch_size * max_num_agents, 3, 3]
    _, matrix = build_matrix(world_agents_t0_pos_exp.reshape(-1, 2), world_agents_t0_yaw_exp.reshape(-1))
    # [batch_size, max_num_agents, 3, 3]
    matrix = matrix.view(list(world_agent_t0_pos_yaw.shape[:2]) + [3, 3])
    # [batch_size * max_num_agents * num_trajectories * num_timesteps, 3, 3]
    matrix = matrix.unsqueeze(2).unsqueeze(2).expand(list(agent_pos.shape[:-1]) + [3, 3]).reshape(-1, 3, 3)
    coords_offset = transform_points(world_agent_pos.reshape(-1, 1, 1, 3), matrix,
                                     avail=torch.ones_like(world_agent_pos.reshape(-1, 1, 1, 3)[..., 0]))
    coords_offset = coords_offset.view_as(world_agent_pos)

    # need to filter per agents of interest (from original prediction evaluation)
    agents_track_ids = data["all_other_agents_track_ids"]
    agents_of_interest = data["all_valid_agents_track_ids"]
    agents_track_ids_mask = torch.zeros_like(agents_track_ids, dtype=torch.bool)
    missing_agents_mask = torch.zeros_like(agents_of_interest, dtype=torch.bool)
    for batch_idx in range(agents_track_ids.shape[0]):
        agents_track_ids_mask[batch_idx] = torch_isin(agents_track_ids[batch_idx], agents_of_interest[batch_idx]) * \
                                           agents_track_ids[batch_idx] != 0
        missing_agents_mask[batch_idx] = ~torch_isin(agents_of_interest[batch_idx], agents_track_ids[batch_idx]) * \
                                         agents_of_interest[batch_idx] != 0
    # we may miss some agents due to the limit cfg["data_generation_params"]["other_agents_num"], we will consider them stationary
    missing_agents_ids = agents_of_interest[missing_agents_mask]
    if torch.any(missing_agents_mask):
        # print(len(missing_agents_ids), missing_agents_ids[missing_agents_ids != 0])
        missing_agents_ids = missing_agents_ids[missing_agents_ids != 0]
        missing_agent_of_interest_ids.append(missing_agents_ids.cpu())
        missing_timestamps = []
        for batch_idx, num_missing_agents in enumerate(missing_agents_mask.sum(-1)):
            missing_timestamps.extend([data["timestamp"][batch_idx]] * num_missing_agents)
        missing_agent_of_interest_timestamp.append(torch.tensor(missing_timestamps))

    # move the valida data to CPU
    relevant_coords_offset = coords_offset[agents_track_ids_mask].cpu()
    traj_confidence = agent_logits[agents_track_ids_mask].cpu()
    relevant_agent_track_ids = agents_track_ids[agents_track_ids_mask].cpu()
    relevant_timestamps = data["timestamp"].unsqueeze(1).expand(agents_track_ids.shape)[agents_track_ids_mask].cpu()

    # add them to the result lists
    future_coords_offsets_pd.append(relevant_coords_offset)
    future_traj_confidence.append(traj_confidence)
    timestamps.append(relevant_timestamps)
    agent_ids.append(relevant_agent_track_ids)


# add the missing agents as stationary
missing_agent_of_interest_ids = torch.cat(missing_agent_of_interest_ids, dim=0)
missing_agent_of_interest_timestamp = torch.cat(missing_agent_of_interest_timestamp, dim=0)
stationary_trajectories = torch.zeros(list(missing_agent_of_interest_ids.shape[:1]) + list(future_coords_offsets_pd[0].shape[1:]))
uniform_probabilities = torch.ones(list(missing_agent_of_interest_ids.shape[:1]) + list(future_traj_confidence[0].shape[1:]))
agent_ids.append(missing_agent_of_interest_ids)
future_coords_offsets_pd.append(stationary_trajectories)
future_traj_confidence.append(uniform_probabilities)
timestamps.append(missing_agent_of_interest_timestamp)

# concatenate all the results in a single np array
future_coords_offsets_pd = torch.cat(future_coords_offsets_pd, dim=0).numpy()
future_traj_confidence = torch.cat(future_traj_confidence, dim=0).softmax(-1).numpy()
timestamps = torch.cat(timestamps, dim=0).numpy().astype(np.int64)
agent_ids = torch.cat(agent_ids, dim=0).numpy().astype(np.int64)  


# let's verify the number of coordinates corresponds to the number of coordinates in the original
assert len(future_coords_offsets_pd == 94694)

print(f"Overall, we missed {len(missing_agent_of_interest_ids)} agents over a total of {94694} agents "
      f"(~{len(missing_agent_of_interest_ids)/94694:.5f}%)")


# -- -- #
pred_path = f"{os.getcwd()}/new_test_pred.csv"

write_pred_csv(pred_path,
               timestamps=timestamps,
               track_ids=agent_ids,
               coords=future_coords_offsets_pd[..., :2],
               confs=future_traj_confidence,
               max_modes=1)
