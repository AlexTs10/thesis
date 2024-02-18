import torch.nn as nn

def mse_loss(model_output, ground_truth, availability_mask):
        
    # Prepare the mask for broadcasting by adding an extra dimension for the xy coordinates
    expanded_mask = availability_mask.unsqueeze(-1)  # This adds a dimension for the xy, resulting in shape (bs, 50, 50, 1)
    
    # Ensure the mask is broadcastable over the xy coordinates
    # By using float() on the mask, you ensure the operations are performed correctly
    model_output_masked = model_output * expanded_mask.float()
    ground_truth_masked = ground_truth * expanded_mask.float()

    # Initialize the MSE Loss criterion
    criterion = nn.MSELoss()
    
    # Calculate the MSE Loss using the masked outputs and ground truths
    loss = criterion(model_output_masked, ground_truth_masked)
    
    return loss.item()



