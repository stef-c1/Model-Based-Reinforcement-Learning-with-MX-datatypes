import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
from qtorch import FloatingPoint,BlockFloatingPoint,FixedPoint
from qtorch.quant import Quantizer, quantizer


#sorts the weights(x) before quant and reconstructs them so they are at the same place as in the begining



x = torch.rand(10,4)
x_orig = x
print(x)
wl = 8
block_size_rows = 4
block_size_cols = 2
rounding="nearest"
log_exp_name="temp"
block_granularity='fine'
doing_mx=1
E=4
M=3
E_bias=7

size_x = x.size()



flag = 0
if len(x.shape)==2:
    x = x.unsqueeze(0)
    flag = 1
if len(x.shape)==1:
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    flag = 2

kh, kv =  block_size_cols, block_size_rows 
sh, sv =  block_size_cols, block_size_rows 

if block_granularity=="vector_row":
    kh, kv = x.size(2) , 1
    sh, sv = x.size(2) , 1
elif block_granularity=="vector_col":
    kh, kv = 1 , x.size(1)
    sh, sv = 1 , x.size(1)

# print(x.shape)
nh, remainder = np.divmod(x.size(2), kh)
nh += bool(remainder)
nv, remainder =  np.divmod(x.size(1), kv)
nv += bool(remainder)

ph, pv = nh*kh - x.size(2), nv*kv - x.size(1)

x_pad = F.pad(x, (0, ph, 0, pv, 0, 0))
x_pad_unfold = x_pad.unfold(1,kv,sv).unfold(2,kh,sh).reshape(-1, kv, kh).float()

# print(x_pad_unfold)s
# print(x_pad_unfold.size())

#sort
x_flat = x_pad_unfold.flatten()
x_sort, indices = x_flat.sort()
x_out = x_sort.unflatten(0, x_pad_unfold.size())
# print(x_pad_unfold.size() == x_out.size())

# print(x_out)
##########QUANT#################
x_q = x_out

#unsort
x_flat2 = x_q.flatten()
x_sort2 = x_flat2.gather(0,indices.argsort())
x_sort3 = x_sort2.gather(0,indices.argsort())
print(x_sort3)
print(x_sort2)
x_out2 = x_sort2.unflatten(0,x_q.size())
# print(x_out2.size() == x_q.size())


x_fold_orig = x_out2.reshape(-1, nv,nh,kv,kh).permute(0,1,3,2,4).reshape(-1, x_pad.shape[1], x_pad.shape[2])    
x_quantized = x_fold_orig[:x.shape[0], :x.shape[1], :x.shape[2]]

if flag == 1:
    x_quantized = x_quantized.squeeze()
    flag = 0
if flag == 2:
    x_quantized = x_quantized.squeeze().squeeze()
    flag = 0




# print(x_quantized==x_orig)




#go back to sorted
x3 = x_pad_unfold.flatten()
x4 = x3[indices]
x5 = x4.unflatten(0,x_pad_unfold.size())
print(x5)
print(x_q)
