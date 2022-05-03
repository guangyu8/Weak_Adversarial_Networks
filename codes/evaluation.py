import numpy as np
import torch

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import truncnorm
from matplotlib import cm

class visual_net():
    
    def __init__(self, params):#, file_path):
        self.mesh_size = params['mesh_size']
        #self.dir= file_path
        
    def show_error(self, iteration, error, dim, name):
        # 画 L_2 relative error vs. iteration 图像的函数
        # This function designed for drawing L_2 relative error vs. iteration
        plt.figure()
        plt.semilogy(iteration, error, color='b')
        plt.xlabel("Iteration", size=20)
        plt.ylabel("Relative error", size=20)        
        plt.tight_layout()
        plt.show()
        #plt.savefig(self.dir+'figure_err/error_iter_%s_%dd.png'%(name, dim))
        plt.close()
        
    def show_error_abs(self, mesh, x_y, z, name, dim):
        # 画pointwise absolute error 图像的函数
        # This function designed for drawing point-wise absolute error
        x= np.ravel(x_y[:,0])
        y= np.ravel(x_y[:,1])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure() 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('x', fontsize=20)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel('y', fontsize=20)
        plt.tight_layout()
        plt.show()
        #plt.savefig(self.dir+'figure_err/error_abs_%s_%dd.png'%(name, dim))
        plt.close()


    def show_u_val(self, mesh, z1, z2, name, i):
        # 画u(x)的函数
        x1, x2 = mesh
        z1= np.reshape(z1, [self.mesh_size, self.mesh_size])
        z2= np.reshape(z2, [self.mesh_size, self.mesh_size])
        #*******************
        fig= plt.figure(figsize=(12,5))
        ax1= fig.add_subplot(1,2,1)
        graph1= ax1.contourf(x1, x2, z1, 10,  cmap= cm.jet)
        fig.colorbar(graph1, ax= ax1)
        #
        ax2= fig.add_subplot(1,2,2)
        graph2= ax2.contourf(x1, x2, z2, 10,  cmap= cm.jet)
        fig.colorbar(graph2, ax= ax2)
        #*******************
        plt.tight_layout()
        plt.show()
        # plt.savefig(self.dir+'figure_%s/%s_val_%d.png'%(name, name, i))        
        plt.close()
        
    def show_v_val(self, mesh, x_y, z, name, i):
        # 画v(x)的函数
        # This function designed for drawing the figure of test function v(x)
        x= np.ravel(x_y[:,0])
        y= np.ravel(x_y[:,1])
        #
        xi,yi = mesh
        zi = griddata((x, y), np.ravel(z), (xi, yi), method='linear')
        plt.figure() 
        plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlim(np.min(xi), np.max(xi))
        plt.xlabel('x', fontsize=20)
        plt.ylim(np.min(yi), np.max(yi))
        plt.ylabel('y', fontsize=20)
        plt.tight_layout()
        plt.show()
        #plt.savefig(self.dir+'figure_%s/%s_%d.png'%(name, name, i))
        plt.close()