import torch


def mock_vectorizer_data(batch_size: int, num_steps: int, num_history: int, num_agents: int, num_lanes: int,
                         num_crosswalks: int, num_points_per_element: int, TYPE_MAX: int) -> dict:
    return {
        "extent": torch.rand(batch_size, 3),
        "type": torch.randint(0, TYPE_MAX, (batch_size,)),
        "world_to_image": torch.rand(batch_size, 3, 3),
        "raster_from_agent": torch.rand(batch_size, 3, 3),
        "raster_from_world": torch.rand(batch_size, 3, 3),
        "agent_from_world": torch.rand(batch_size, 3, 3),
        "world_from_agent": torch.rand(batch_size, 3, 3),
        "target_positions": torch.rand(batch_size, num_steps, 2),
        "target_yaws": torch.rand(batch_size, num_steps, 1),
        "target_extents": torch.rand(batch_size, num_steps, 2),
        "target_availabilities": torch.rand(batch_size, num_steps) > 0.5,
        "history_positions": torch.rand(batch_size, num_history + 1, 2),
        "history_yaws": torch.rand(batch_size, num_history + 1, 1),
        "history_extents": torch.rand(batch_size, num_history + 1, 2),
        "history_availabilities": torch.rand(batch_size, num_history + 1) > 0.5,
        "centroid": torch.rand(batch_size, 2),
        "yaw": torch.rand(batch_size),
        "speed": torch.rand(batch_size),
        "all_other_agents_history_positions": torch.rand(batch_size, num_agents, num_history + 1, 2),
        "all_other_agents_history_yaws": torch.rand(batch_size, num_agents, num_history + 1, 1),
        "all_other_agents_history_extents": torch.rand(batch_size, num_agents, num_history + 1, 2),
        "all_other_agents_history_availability": torch.rand(batch_size, num_agents, num_history + 1) > 0.5,
        "all_other_agents_future_positions": torch.rand(batch_size, num_agents, num_steps, 2),
        "all_other_agents_future_yaws": torch.rand(batch_size, num_agents, num_steps, 1),
        "all_other_agents_future_availability": torch.rand(batch_size, num_agents, num_steps) > 0.5,
        "all_other_agents_types": torch.randint(0, TYPE_MAX, (batch_size, num_agents)),
        "agent_trajectory_polyline": torch.rand(batch_size, num_history + 1, 3),
        "agent_polyline_availability": torch.rand(batch_size, num_history + 1) > 0.5,
        "other_agents_polyline": torch.rand(batch_size, num_agents, num_history + 1, 3),
        "other_agents_polyline_availability": torch.rand(batch_size, num_agents, num_history + 1) > 0.5,
        "lanes": torch.rand(batch_size, num_lanes, num_points_per_element, 3),
        "lanes_availabilities": torch.rand(batch_size, num_lanes, num_points_per_element) > 0.5,
        "lanes_mid": torch.rand(batch_size, num_lanes, num_points_per_element, 3),
        "lanes_mid_availabilities": torch.rand(batch_size, num_lanes, num_points_per_element) > 0.5,
        "crosswalks": torch.rand(batch_size, num_crosswalks, num_points_per_element, 3),
        "crosswalks_availabilities": torch.rand(batch_size, num_crosswalks, num_points_per_element) > 0.5,
        "scene_index": torch.rand(batch_size),
        "host_id": torch.rand(batch_size, 70),
        "timestamp": torch.rand(batch_size),
        "track_id": torch.rand(batch_size),
    }
