import torch

def temporal_split(data, train_end=34, val_end=40):
    labeled_mask = data.y != -1
    train_mask = labeled_mask & (data.timestep <= train_end)
    val_mask = labeled_mask & (data.timestep > train_end) & (data.timestep <= val_end)
    test_mask = labeled_mask & (data.timestep > val_end)
    return train_mask, val_mask, test_mask