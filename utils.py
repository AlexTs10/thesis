from l5kit.planning.vectorized.common import pad_avail, pad_points
from l5kit.evaluation import create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.data import LocalDataManager
import os 
import torch 
import numpy as np 

def val_dataset_setup(cfg):
    dm = LocalDataManager(None)
    num_frames_to_chop = 100
    eval_cfg = cfg["val_data_loader"]
    eval_base_path = os.path.join(os.environ["L5KIT_DATA_FOLDER"], f"{eval_cfg['key'].split('.')[0]}_chopped_100")
    if not os.path.exists(eval_base_path):
        eval_base_path = create_chopped_dataset(
            dm.require(eval_cfg["key"]), 
            cfg["raster_params"]["filter_agents_threshold"], 
            num_frames_to_chop, 
            cfg["model_params"]["future_num_frames"], 
            MIN_FUTURE_STEPS)
    

def criterion(gt, pred, confidences, avails, num_modes=3):
    
    bs, future_len, num_coords = gt.shape
    
    # Assertions and checks (some are omitted for brevity)
    assert gt.shape == (bs, future_len, num_coords)
    assert pred.shape == (bs, num_modes, future_len, num_coords)
    assert confidences.shape == (bs, num_modes)
    assert avails.shape == (bs, future_len)
    assert np.allclose(torch.sum(confidences, axis=1).detach().numpy(), 1), "confidences should sum to 1"
    
    # Further processing as in the original function...
    # Expand dimensions for gt: shape becomes (bs, 1, future_len, num_coords)
    gt = gt.unsqueeze(1)
    # Expand dimensions for avails: shape becomes (bs, 1, future_len, 1)
    avails = avails.unsqueeze(1).unsqueeze(-1)
    
    # Calculate squared error, element-wise multiplied by avails # Sum over the last dimension (coordinates)
    error = torch.sum((((gt - pred) * avails) ** 2), dim=-1)
    
    # Use torch's equivalent of np.errstate
    #with torch.no_grad():
    # When confidence is 0, log goes to -inf, but we're fine with it
    error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # Reduce time
    
    # Use max aggregator on modes for numerical stability
    max_value = torch.max(error, dim=-1)[0]  # Get the max along the mode dimension
    
    # Subtract max_value for numerical stability and sum across modes
    error = -torch.log(torch.sum(torch.exp(error - max_value.unsqueeze(-1)), dim=-1)) - max_value
    return error.mean()

def preprocess(data_batch):

    # normalization buffers
    agent_std =  torch.tensor([1.6919, 0.0365, 0.0218])
    other_agent_std = torch.tensor([33.2631, 21.3976, 1.5490])

    # ==== LANES ====
    # batch size x num lanes x num vectors x num features
    polyline_keys = ["lanes_mid", "crosswalks"]
    polyline_keys += ["lanes"]
    avail_keys = [f"{k}_availabilities" for k in polyline_keys]
    
    max_num_vectors = max([data_batch[key].shape[-2] for key in polyline_keys])
    
    map_polys = torch.cat([pad_points(data_batch[key], max_num_vectors) for key in polyline_keys], dim=1)
    map_polys[..., -1].fill_(0)
    # batch size x num lanes x num vectors
    map_availabilities = torch.cat([pad_avail(data_batch[key], max_num_vectors) for key in avail_keys], dim=1)
    
    # ==== AGENTS ====
    # batch_size x (1 + M) x seq len x self._vector_length
    agents_polys = torch.cat(
        [data_batch["agent_trajectory_polyline"].unsqueeze(1), data_batch["other_agents_polyline"]], dim=1
    )
    # batch_size x (1 + M) x num vectors x self._vector_length
    agents_polys = pad_points(agents_polys, max_num_vectors)
    
    # batch_size x (1 + M) x seq len
    agents_availabilities = torch.cat([data_batch["agent_polyline_availability"].unsqueeze(1),
                                        data_batch["other_agents_polyline_availability"],],
                                      dim=1,)
    
    # batch_size x (1 + M) x num vectors
    agents_availabilities = pad_avail(agents_availabilities, max_num_vectors)
    
    # Standardize inputs
    agents_polys_feats = torch.cat(
        [agents_polys[:, :1] / agent_std, agents_polys[:, 1:] / other_agent_std], dim=1
    )
    static_polys_feats = map_polys / other_agent_std
    
    all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1) # bs, M, T, Emb
    all_polys = all_polys.permute(0, 2, 1, 3) # bs, T, M, Emb
    all_avail = torch.cat([agents_availabilities, map_availabilities], dim=1)

    # [batch_size, num_elements, max_num_points]
    invalid_mask = ~all_avail 
    invalid_polys = invalid_mask.all(-1)
    
    return all_polys, invalid_polys 