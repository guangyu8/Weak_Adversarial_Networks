import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

class net_a(torch.nn.Module):
    def __init__(self, width, depth):
        super(net_a, self).__init__()
        self.depth = depth
        self.linear = [torch.nn.Linear(2, width)]
        self.linear.append(torch.nn.Linear(width, width))
        for i in range(depth):
            self.linear.append(torch.nn.Linear(width, width))        
        self.linear.append(torch.nn.Linear(width, 1))

        self.linear = torch.nn.ModuleList(self.linear)
    
    def forward(self, x):
        y = x
        y = F.tanh(self.linear[0](y))
        y = F.tanh(self.linear[1](y))
        for i in range(2, self.depth + 2):
            if i%2 == 0:
                y = F.sigmoid(self.linear[i](y))
                #y = torch.tanh(self.linear[i](y))
            else:
                y = F.tanh(self.linear[i](y))

        return F.sigmoid(self.linear[-1](y))


class net_u(torch.nn.Module):
    def __init__(self, width, depth):
        super(net_u, self).__init__()
        self.depth = depth
        self.linear = [torch.nn.Linear(2, width)]
        self.linear.append(torch.nn.Linear(width, width))
        for i in range(depth):
            self.linear.append(torch.nn.Linear(width, width))        
        self.linear.append(torch.nn.Linear(width, 1))

        self.linear = torch.nn.ModuleList(self.linear)
    
    def forward(self, x):
        y = x
        y = F.tanh(self.linear[0](y))
        y = F.tanh(self.linear[1](y))
        for i in range(2, self.depth + 2):
            if i%2 == 0:
                y = F.softplus(self.linear[i](y))
                #y = torch.tanh(self.linear[i](y))
            else:
                y = torch.sin(self.linear[i](y))

        return self.linear[-1](y)
    
class net_v(torch.nn.Module):
    def __init__(self, width, depth):
        super(net_v, self).__init__()
        self.depth = depth
        self.linear = [torch.nn.Linear(2, width)]
        self.linear.append(torch.nn.Linear(width, width))
        for i in range(depth):
            self.linear.append(torch.nn.Linear(width, width))        
        self.linear.append(torch.nn.Linear(width, 1))

        self.linear = torch.nn.ModuleList(self.linear)
    
    def forward(self, x):
        y = x
        y = F.tanh(self.linear[0](y))
        y = F.tanh(self.linear[1](y))
        for i in range(2, self.depth + 2):
            if i%2 == 0:
                y = torch.sin(self.linear[i](y))
                #y = torch.tanh(self.linear[i](y))
            else:
                y = torch.sin(self.linear[i](y))

        return self.linear[-1](y)
    
def fun_w(x):
    low,up,dim = 0,1,2
    I1= 0.110987    
    #**************************************************    
    h_len= (up-low)/2.0
    x_scaled = (x - low-h_len)/h_len
    #************************************************
    supp_x= torch.greater(1-torch.abs(x_scaled), 0)
    z_x= torch.where(supp_x, torch.exp(1/(torch.pow(x_scaled, 2)-1))/I1, 
                        torch.zeros_like(x_scaled))
    
    #***************************************************
    w_val= 1
    w= w_val*torch.multiply(z_x[:,0], z_x[:,1])
    # w = torch.reshape(w, (len(w),1))
    w = torch.reshape(w, [-1,1])
    dw = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True)
    dw = dw[0]  # to get tensor from tuple
    dw= torch.where(torch.isnan(dw), torch.zeros_like(dw), dw)
    return w, dw

def fun_g(x, n_vec, params):
    coef = (params['dim'] - 1) * params['la']**2/2
    u_val = torch.exp(coef*x[:,0]*(x[:,0]-1))
    for i in range(params['dim']-1):
        u_val= torch.multiply(u_val, torch.sin(params['la']*x[:,i+1]))
    u_val = torch.reshape(u_val, [-1, 1])
    
    du = torch.autograd.grad(u_val, x, grad_outputs=torch.ones_like(u_val), retain_graph=True)[0]
    du = torch.where(torch.isnan(du), torch.zeros_like(du), du)
    g_obv = torch.sum(torch.multiply(du, n_vec), 1, keepdim=True).reshape(-1, 1)
    return (u_val, du, g_obv)

def grad_u(u, x, test_mode=False):
    x_list = torch.tensor_split(x, x.shape[1], 1)
    x = torch.cat(x_list, 1)

    u_val = u.forward(x)
    du = autograd.grad(u_val, x, torch.ones_like(u_val), retain_graph=True, create_graph=True)[0]    
    du_list = torch.tensor_split(du, du.shape[1], 1)  
    if test_mode:  
        div_du = torch.zeros_like(du_list[0])
        for i in range(du.shape[1]):
            div_du_i = autograd.grad(du_list[i], x_list[i], torch.ones_like(du_list[i]), retain_graph=True, create_graph=False)[0]
            div_du = div_du+div_du_i
    else:
        div_du = None
    return u_val, du, div_du

def grad_v(v, x):
    v_val = v.forward(x)
    dv = autograd.grad(v_val, x, torch.ones_like(v_val), retain_graph=True, create_graph=False)[0]    
    return v_val, dv


def grad_ua(u, a, x):
    x_list = torch.tensor_split(x, x.shape[1], 1)
    x = torch.cat(x_list, 1)

    u_val = u(x)
    a_val = a(x)
    du = autograd.grad(u_val, x, torch.ones_like(u_val), retain_graph=True, create_graph=True)[0]   
    dua = torch.multiply(du, a_val)
    dua_list = torch.tensor_split(dua, dua.shape[1], 1)  
    div_dua = torch.zeros_like(dua_list[0])
    for i in range(dua.shape[1]):
        div_dua_i = autograd.grad(dua_list[i], x_list[i], torch.ones_like(dua_list[i]), retain_graph=True, create_graph=True)[0]
        div_dua = div_dua+div_dua_i

    return u_val, du, a_val, div_dua