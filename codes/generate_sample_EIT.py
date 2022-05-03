import numpy as np
import torch

def sample_train(params):
    low, up= params['low'], params['up']
    dim = params['dim']; dm_size = params['dm_size']; bd_size = params['bd_size']
    # generate taining data
    #********************************************************
    # collocation points in domain
    x_dm= np.random.uniform(low, up, [dm_size, dim])
    #*********************************************************
    # The value of f(x)
    x= np.reshape(x_dm[:,0], [-1,1])
    f_dm= -2*np.ones([np.shape(x_dm)[0], 1])
    #*********************************************************
    # collocation points on boundary
    x_bd_list=[]; n_vector_list=[]
    for i in range(dim):
        x_bound= np.random.uniform(low, up, [bd_size, dim])
        x_bound[:,i]= up
        x_bd_list.append(x_bound)
        x_bound= np.random.uniform(low, up, [bd_size, dim])
        x_bound[:,i]= low
        x_bd_list.append(x_bound)
        
        n_vector= np.zeros_like(x_bound)
        n_vector[:,i]=1
        n_vector_list.append(n_vector)
        n_vector= np.zeros_like(x_bound)
        n_vector[:,i]=-1
        n_vector_list.append(n_vector)
    x_bd= np.concatenate(x_bd_list, axis=0)   
    n_vector= np.concatenate(n_vector_list, 0)
    #***********************************************************
    # observation of u(x) on boundary
    #x = np.reshape(x_bd[:,0], [-1,1])
    coef = (dim - 1) * params['la']**2/2
    #u_bd= np.where(x<=0.5, x**2, (x-1)**2)
    u_bd= np.exp(coef*x_bd[:,0]*(x_bd[:,0]-1))
    for i in range(dim-1):
        u_bd= np.multiply(u_bd, np.sin(params['la']*x_bd[:,i+1]))
    u_bd= np.reshape(u_bd, [-1, 1])
    # observation of a(x) on boundary
    a_bd= np.exp(coef*x_bd[:,0]*(1-x_bd[:,0]))
    a_bd= np.reshape(a_bd/params['la'], [-1,1])
    #*********************************************************
    int_dm= (up-low)**dim
    #
    train_dict={}
    x_dm= np.float32(x_dm); train_dict['x_dm']= x_dm
    x_bd= np.float32(x_bd); train_dict['x_bd']= x_bd
    int_dm= np.float32(int_dm); train_dict['int_dm']= int_dm
    #f_dm= np.float32(f_dm); train_dict['f_val']= f_dm
    a_bd= np.float32(a_bd); train_dict['a_bd']= a_bd
    u_bd= np.float32(u_bd); train_dict['u_bd']= u_bd
    n_vector= np.float32(n_vector); train_dict['n_vector']= n_vector
    return(train_dict)


def sample_test(params):
    # testing data
    low, up= params['low'], params['up']
    dim = params['dim']; mesh_size = params['mesh_size']
    #**********************************************************
    # generate meshgrid in the domain
    x_mesh= np.linspace(low, up, mesh_size)
    mesh= np.meshgrid(x_mesh, x_mesh)
    #
    x1_dm= np.reshape(mesh[0], [-1,1])
    x2_dm= np.reshape(mesh[1], [-1,1])
    #
    x3_dm= np.random.uniform(low, up, [mesh_size*mesh_size, dim-2])
    x_dm= np.concatenate([x1_dm, x2_dm, x3_dm], axis=1)
    x4_dm= np.zeros([mesh_size*mesh_size, dim-2])
    x_draw_dm= np.concatenate([x1_dm, x2_dm, x4_dm], axis=1)
    #***********************************************************
    # The exact u(x)
    #u_dm= np.where(x1_dm<=0.5, x1_dm**2, (x1_dm-1)**2)
    #u_draw_dm= np.where(x1_dm<=0.5, x1_dm**2, (x1_dm-1)**2)
    coef = (dim - 1) * params['la']**2/2
    u_dm= np.exp(coef*x_dm[:,0]*(x_dm[:,0]-1))
    u_draw_dm= np.exp(coef*x_draw_dm[:,0]*(x_draw_dm[:,0]-1))
    for i in range(dim-1):
        u_dm= np.multiply(u_dm, np.sin(params['la']*x_dm[:,i+1]))
        u_draw_dm= np.multiply(u_draw_dm, np.sin(params['la']*x_draw_dm[:,i+1]))
    u_dm= np.reshape(u_dm, [-1, 1])
    u_draw_dm= np.reshape(u_draw_dm, [-1, 1])
    a_dm= np.exp(coef*x_dm[:,0]*(1-x_dm[:,0]))
    a_dm= np.reshape(a_dm/params['la'], [-1,1])  
    a_draw_dm= np.exp(coef*x_draw_dm[:,0]*(1-x_draw_dm[:,0]))
    a_draw_dm= np.reshape(a_draw_dm/params['la'], [-1,1])
    #***********************************************************
    test_dict={}; test_dict['mesh']= mesh
    x_dm= np.float32(x_dm); test_dict['test_x']= x_dm
    x_draw_dm= np.float32(x_draw_dm); test_dict['draw_x']= x_draw_dm 
    u_dm= np.float32(u_dm); test_dict['test_u']= u_dm 
    u_draw_dm= np.float32(u_draw_dm); test_dict['draw_u']= u_draw_dm
    a_dm= np.float32(a_dm); test_dict['test_a']= a_dm 
    a_draw_dm= np.float32(a_draw_dm); test_dict['draw_a']= a_draw_dm
    return(test_dict)