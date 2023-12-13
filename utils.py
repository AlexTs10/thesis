from l5kit.planning.vectorized.common import pad_avail, pad_points
import torch 

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
    
    all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1)
    all_avail = torch.cat([agents_availabilities, map_availabilities], dim=1)

    # [batch_size, num_elements, max_num_points]
    invalid_mask = ~all_avail 
    invalid_polys = invalid_mask.all(-1)

    bs, num_obj, num_time, num_coord =  all_polys.shape
    invalid_polys = invalid_polys.unsqueeze(-1).expand(-1, -1, num_time).reshape(bs, num_obj*num_time)
    
    return all_polys.reshape(bs, num_obj*num_time, num_coord), invalid_polys 

