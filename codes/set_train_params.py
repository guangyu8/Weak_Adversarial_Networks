import numpy as np
import torch

def set_params(dim=2):
    params = dict()
    
    params['dim'] = dim  # dimension of the data
    params['dm_size'] = 100000   # sample points inside domain
    params['bd_size'] = 50  # sample points at boundary
    
    params['mesh_size'] = 50 # mesh size for test data set
    params['beta'] = 10000 
    
    params['low'], params['up'] = 0., 1.  # square boundary location
    params['la'] = np.pi
    
    params['u_steps'] = 1   # training step
    params['v_steps'] = 1
    return params


def set_params_pinn(dim=2):
    params = dict()
    
    params['dim'] = dim  # dimension of the data
    params['dm_size'] = 100000   # sample points inside domain
    params['bd_size'] = 50  # sample points at boundary
    
    params['mesh_size'] = 50 # mesh size for test data set
    params['beta'] = 200000 
    
    params['low'], params['up'] = 0., 1.  # square boundary location
    params['la'] = np.pi
    
    params['u_steps'] = 1   # training step    
    params['a_steps'] = 1
    return params

def set_params_drm(dim=2):
    params = dict()
    
    params['dim'] = dim  # dimension of the data
    params['dm_size'] = 100000   # sample points inside domain
    params['bd_size'] = 50  # sample points at boundary
    
    params['mesh_size'] = 50 # mesh size for test data set
    params['beta'] = 200000000 
    
    params['low'], params['up'] = 0., 1.  # square boundary location
    params['la'] = np.pi
    
    params['u_steps'] = 1   # training step    
    params['a_steps'] = 1
    return params