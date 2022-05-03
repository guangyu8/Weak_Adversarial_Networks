import numpy as np
import torch

def tensor_to_numpy(z):
    if z.device=="cpu":
        return z.detach().numpy()
    else:
        return z.cpu().detach().numpy()