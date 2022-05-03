import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from model import fun_w, fun_g, grad_v, grad_u, grad_ua
from others import tensor_to_numpy

def iwan_eit_loss(net_u, net_a, net_v, x, x_bd, n_vec, u_bd, a_bd, params):
    device = x.device
    w_val, dw = fun_w(x)        
    #f_val = torch.tensor(f_val).to(device)
    #u_bd = torch.tensor(u_bd).to(device)
    
    u_val, du, div_du = grad_u(net_u, x)
    u_bd_pred, du_bd, _ = grad_u(net_u, x_bd)
    a_val = net_a(x)
    a_bd_pred = net_a(x_bd)
    v_val, dv = grad_v(net_v, x)
    w_val = torch.Tensor(tensor_to_numpy(w_val)).to(device)
    dw = torch.Tensor(tensor_to_numpy(dw)).to(device)
    wv_val= torch.multiply(w_val, v_val)
    #
    dudw_val= torch.sum(torch.multiply(du, dw), 1, keepdim=True)    
    #
    dudv_val= torch.sum(torch.multiply(du, dv), 1, keepdim=True)    
    #
    dudwv_val= torch.add(torch.multiply(v_val, dudw_val),
                        torch.multiply(w_val, dudv_val)) 
    
    _, _, g_obv = fun_g(x_bd, n_vec, params)
    g_val= torch.sum(torch.multiply(du_bd, n_vec), 1, keepdim=True)
    # defining loss_u
    int_r1 = torch.mean(torch.multiply(a_val, dudwv_val))
    test_norm= torch.mean(wv_val**2)
    loss_int= 10*torch.square(int_r1) / test_norm
    #*********************************************************
    loss_u_bd = torch.mean(torch.abs(u_bd_pred - u_bd))
    loss_g_bd = torch.mean(torch.abs(g_val - g_obv))
    loss_a_bd = torch.mean(torch.abs(a_bd_pred - a_bd))
    loss_total = params['beta'] * (loss_u_bd + loss_g_bd + loss_a_bd) + loss_int
    ##########################################################
    #point_dist= dudwv_val-torch.multiply(f_val, wv_val)    
    #int_1= torch.mean(point_dist)
    #loss_int= beta_int*torch.square(int_1) / test_norm
    #
    #point_dist= dudw_val-torch.multiply(f_val, w_val)
    #int_1= torch.mean(point_dist)
    #loss_intw= beta_intw*torch.square(int_1)/ w_norm
    #**********************************************************
    #loss_bd= torch.mean(torch.abs(u_bd_pred- u_bd))    
    #loss_u= beta_bd*loss_bd + beta_int*loss_int

    # defining loss_v
    loss_v = - torch.log(loss_int)
    
    return loss_total, loss_v



def iPINN_eit_loss(net_u, net_a, x, x_bd, n_vec, u_bd, a_bd, params):
    device = x.device
    _,_,_,div_dua = grad_ua(net_u, net_a, x)

    a_bd_pred = net_a(x_bd)
    u_bd_pred, du_bd, _ = grad_u(net_u, x_bd)  
    _, _, g_obv = fun_g(x_bd, n_vec, params)
    g_val= torch.sum(torch.multiply(du_bd, n_vec), 1, keepdim=True)
        
    inner_diff = div_dua
    inner_loss = torch.mean(inner_diff**2)

    loss_u_bd = torch.mean(abs(u_bd_pred - u_bd))
    loss_g_bd = torch.mean(abs(g_val - g_obv))
    loss_a_bd = torch.mean(abs(a_bd_pred - a_bd))

    loss_total = params['beta'] * (loss_u_bd + loss_g_bd + loss_a_bd) + inner_loss

    return loss_total


def idrm_eit_loss(net_u, net_a, x, x_bd, n_vec, u_bd, a_bd, params):
    device = x.device

    u_val, du, _ = grad_u(net_u, x)
    a_val = net_a(x)
    
    a_bd_pred = net_a(x_bd)
    u_bd_pred, du_bd, _ = grad_u(net_u, x_bd)  
    _, _, g_obv = fun_g(x_bd, n_vec, params)
    g_val= torch.sum(torch.multiply(du_bd, n_vec), 1, keepdim=True)
                    
    inner_energy = torch.multiply(torch.sum(du**2, 1, keepdim = True), a_val)
    inner_loss = torch.mean(inner_energy)

    loss_u_bd = torch.mean(abs(u_bd_pred - u_bd))
    loss_g_bd = torch.mean(abs(g_val - g_obv))
    loss_a_bd = torch.mean(abs(a_bd_pred - a_bd))

    loss_total = params['beta'] * (loss_u_bd + loss_g_bd + loss_a_bd) + inner_loss

    return loss_total