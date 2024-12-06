
import torch
import torch.nn as nn
import os
import copy
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from config_mx_mbrl.cartpole_qpy2 import CONFIG_MODULE as cartpole_config 
from config_mx_mbrl.pusher_qpy2 import CONFIG_MODULE as pusher_config 
from config_mx_mbrl.reacher_qpy2 import CONFIG_MODULE as reacher_config 
from config_mx_mbrl.halfcheetah_qpy2 import CONFIG_MODULE as halfcheetah_config 


from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize

import json
import ast
from pathlib import Path
import pickle
import time
import math


a = torch.tensor([[1.0,-5.0],[10.0,-2.0]])
print(a.mean(-1))



# # MomentumOptimizer 
# class MomentumOptimizer(torch.optim.Optimizer): 
      
#     # Init Method: 
#     def __init__(self, params, lr=1e-3, momentum=0.9): 
#         super(MomentumOptimizer, self).__init__(params, defaults={'lr': lr}) 
#         self.momentum = momentum 
#         self.state = dict() 
#         for group in self.param_groups: 
#             for p in group['params']: 
#                 self.state[p] = dict(mom=torch.zeros_like(p.data)) 
      
#     # Step Method 
#     def step(self): 
#         for group in self.param_groups: 
#             for p in group['params']: 
#                 if p not in self.state: 
#                     self.state[p] = dict(mom=torch.zeros_like(p.data)) 
#                 mom = self.state[p]['mom'] 
#                 mom = self.momentum * mom - group['lr'] * p.grad.data 
#                 p.data += mom


# class AdamOptimizer(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, spike_data_dict={'grad':0, 'm':0, 'lr':0})
#         super(AdamOptimizer, self).__init__(params, defaults)

#     def step(self, closure=None, spike=0):
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group['params']: #p is a set of weights or biases of a layer
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 state = self.state[p]

#                 # Initialize state if first time
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['exp_avg'] = torch.zeros_like(p.data)
#                     state['exp_avg_sq'] = torch.zeros_like(p.data)

#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 beta1, beta2 = group['betas']
#                 eps = group['eps']
#                 lr = group['lr']
#                 weight_decay = group['weight_decay']

#                 # Update step count
#                 state['step'] += 1

#                 # Apply weight decay if specified
#                 if weight_decay != 0:
#                     grad = grad.add(weight_decay, p.data)

#                 # Compute the running averages of the gradient and squared gradient
#                 exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

#                 # Bias correction
#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']

#                 exp_avg_hat = exp_avg / bias_correction1
#                 exp_avg_sq_hat = exp_avg_sq / bias_correction2
#                 print(exp_avg_sq_hat)

#                 # Update parameters
#                 p.data.addcdiv_(exp_avg_hat, exp_avg_sq_hat.sqrt().add_(eps), value=-lr)
                    
#         return loss




# # Define a simple model 
# model = nn.Linear(2, 1) 
# initial_weight = copy.deepcopy(model.weight)
# initial_bias = copy.deepcopy(model.bias)
  
# # Define a loss function 
# criterion = nn.MSELoss() 
  

# # Generate some random data 
# X = torch.randn(100, 2) 
# y = torch.randn(100, 1)

# loss_lst = [[*range(0,2500,100)],[],[],[]]
# spike = 0
# spike_dict_data = {}

# batch_size = 25
# nb_batches = int(math.ceil(100/batch_size))
# # Define the optimizer 
# for j in range(1,2):
#     if (j==0):
#         optimizer = MomentumOptimizer(model.parameters(), lr=1e-3, momentum=0.9)
#     elif (j==1):
#         optimizer = AdamOptimizer(model.parameters(), lr=1e-3)
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)

#     prev_prev_epoch_batch_loss = []
#     prev_epoch_batch_loss = []
#     batch_loss = []

#     prev_prev_epoch_batch_m= []
#     prev_epoch_batch_m = []
#     batch_m = []

#     prev_prev_epoch_batch_v = []
#     prev_epoch_batch_v = []
#     batch_v = []

#     prev_prev_epoch_batch_grads = []
#     prev_epoch_batch_grads = []
#     batch_grads = []
#     # Training loop 
#     for i in range(2500): 
#         prev_prev_epoch_batch_loss = prev_epoch_batch_loss
#         prev_epoch_batch_loss = copy.deepcopy(batch_loss)
#         batch_loss = []

#         prev_prev_epoch_batch_m = prev_epoch_batch_m
#         prev_epoch_batch_m = copy.deepcopy(batch_m)
#         batch_m = []

#         prev_prev_epoch_batch_v = prev_epoch_batch_v
#         prev_epoch_batch_v = copy.deepcopy(batch_v)
#         batch_v = []

#         prev_prev_epoch_batch_grads = prev_epoch_batch_grads
#         prev_epoch_batch_grads = copy.deepcopy(batch_grads)
#         batch_grads = []
#         for b in range(nb_batches):
#             X_batch = X[b*batch_size: (b+1)*batch_size]

#             optimizer.zero_grad() 
#             y_pred = model(X_batch)
#             loss = criterion(y_pred, y[b*batch_size: (b+1)*batch_size]) 
#             print(loss.size())
#             time.sleep(5)

#             batch_loss.append(loss.cpu().data)
            


#             loss.backward() 
#             optimizer.step(spike=spike)


#             lst_m = []
#             lst_v = []
#             lst_g = []
#             for group in optimizer.param_groups:
#                 for p in group['params']:
#                     print(optimizer.state[p])
#                     time.sleep(5)
#                     lst_m.append(optimizer.state[p]['exp_avg'].clone().detach())
#                     lst_v.append(optimizer.state[p]['exp_avg_sq'].clone().detach())
#                     lst_g.append(copy.deepcopy(p.grad.data))

                    
#                     # print(optimizer.state[p]) #for m and v
#                     # print(p.grad.data) #for grad
#             batch_m.append(copy.deepcopy(lst_m))
#             batch_v.append(copy.deepcopy(lst_v))
#             batch_grads.append(copy.deepcopy(lst_g))



#         # Plot losses 
#         if i%100 ==0:
#             loss_lst[j+1].append(np.mean(batch_loss))

#         if ((i%50 == 0) & (i < 500)):
#             spike = 1
#         else:
#             spike = 0         

#         if (spike):
#             spike_dict_data[str(i)] = {'prev_prev_epoch_batch_loss':  prev_prev_epoch_batch_loss,   'prev_epoch_batch_loss': prev_epoch_batch_loss,  'current_epoch_batch_loss' : batch_loss, 
#                                        'prev_prev_epoch_batch_m'   :  prev_prev_epoch_batch_m,      'prev_epoch_batch_m':    prev_epoch_batch_m,     'current_epoch_batch_m':     batch_m, 
#                                        'prev_prev_epoch_batch_v'   :  prev_prev_epoch_batch_v,      'prev_epoch_batch_v':    prev_epoch_batch_v,     'current_epoch_batch_v':     batch_v, 
#                                        'prev_prev_epoch_batch_grads': prev_prev_epoch_batch_grads, 'prev_epoch_batch_grads': prev_epoch_batch_grads, 'current_epoch_batch_grads': batch_grads}


#     #set model back to initial weights
#     # print(model.weight)
#     # print('init w: '+str(initial_weight))
#     model.weight = initial_weight
#     model.bias = initial_bias
#     initial_weight = copy.deepcopy(initial_weight)
#     initial_bias = copy.deepcopy(initial_bias)


# plt.title('Losses over iterations') 
# plt.xlabel('iterations') 
# plt.ylabel('Losses') 
# #plt.plot(loss_lst[0],loss_lst[1], 'ro-')
# plt.plot(loss_lst[0],loss_lst[2], 'go-')
# #plt.plot(loss_lst[0],loss_lst[3], 'bo-')
# plt.savefig('testfig.png')


# # for key in spike_dict_data.keys():
# key = '50'
# print('epoch: '+str(key))
# print(spike_dict_data[key]['current_epoch_batch_m'])
# len = len(spike_dict_data[key]['current_epoch_batch_m'])
# print(len)
# # for i in range(len):
#     # print(spike_dict_data[key]['current_epoch_batch_m'][i])
#     # print(len(spike_dict_data[key]['current_epoch_batch_m'][i]))


# print(spike_dict_data[key]['current_epoch_batch_v'])
# print(spike_dict_data[key]['current_epoch_batch_grads'])