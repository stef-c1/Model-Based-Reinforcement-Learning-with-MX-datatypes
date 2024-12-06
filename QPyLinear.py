# Linear layer


import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
from qtorch import FloatingPoint,BlockFloatingPoint,FixedPoint
from qtorch.quant import Quantizer, quantizer

f_linear = F.linear
torch_matmul = torch.matmul

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def mx_quant(x, mantissa_len, el_exp_len, el_exp_bias, shared_exp_len=8, dim=-1):#dim only crudely added
    size = x.size()
    quantized_block = torch.zeros(size=size)
    shared_exp_value_block = torch.zeros(size[0])


    e_max_elem = 2**(el_exp_len)-1-1-el_exp_bias #exponent of the largest normal number possible in the element data format; ((2^(Exp_bits)-1)-1)-Exp_bias #first -1 is from representing 0 in the exponent bits (used for subnormal numbers); second -1 is because a full exponent means infinity
    #find shared exponent:
    intermediate_max = torch.max(x.abs(),2)
    max_elems = torch.max(intermediate_max[0].abs(),1)
    shared_exp_block = (torch.floor(torch.log2(max_elems[0]))-e_max_elem)
    # shared_exp_value_block = torch.where(shared_exp_block <= 2**(shared_exp_len)-1, 2**shared_exp_block, 2**(2**shared_exp_len-1)) #not really needed as low chance would go over 2**(2**8), removed for speed reasons
    shared_exp_value_block = 2**shared_exp_block #shared exp bias is already in shared_exp_block as this can be a negative number too

    #quantize to element data format
    scaled_x = x/(shared_exp_value_block.unsqueeze(-1).unsqueeze(-1))
    quantized_block = float_quantize(scaled_x, el_exp_len, mantissa_len, rounding='nearest') # BIAS? -> automatically done in float_quantize

    z = (shared_exp_value_block * quantized_block.T).T
    return z


# Block quantization with tile dimension inputs (Reshape to tiles with padding and quantize and lastly rehape back to original)
def block_quantizer_fine(x, wl = 8, block_size_rows = 4, block_size_cols = 4, rounding="nearest", log_exp_name="temp", block_granularity='fine', doing_mx=1, E=4, M=3, E_bias=7):    
    flag = 0
    
    if block_granularity=="large":
        return block_quantize(x, wl=wl, rounding=rounding, dim=-1) # dim=-1 is for complete tensor

    if len(x.shape)==2:
        x = x.unsqueeze(0)
        flag = 1
    if len(x.shape)==1:
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        flag = 2

      
    blk_quantizer = lambda x: block_quantize(x, wl=wl, rounding=rounding, dim=0)

    mx_quantizer = lambda x: mx_quant(x, M, E, E_bias, dim=0)
    
    # Rearranging large 3D matrix to 3D tensor with blocksize (square) as H x W. 3rd dimension is to complete patches in larger matrix
    
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
    
    ## Insert block quantization with dimension=0 
    if (doing_mx == 1):#BSExMy
        x_q = mx_quantizer(x_pad_unfold)
    elif (doing_mx == 0):#BSINT8
        x_q = blk_quantizer(x_pad_unfold)
    else: #float32
        x_q = x_pad_unfold

    x_fold_orig = x_q.reshape(-1, nv,nh,kv,kh).permute(0,1,3,2,4).reshape(-1, x_pad.shape[1], x_pad.shape[2])    
    x_quantized = x_fold_orig[:x.shape[0], :x.shape[1], :x.shape[2]]
    
    if flag == 1:
        x_quantized = x_quantized.squeeze()
        flag = 0
    if flag == 2:
        x_quantized = x_quantized.squeeze().squeeze()
        flag = 0
        
    return x_quantized

class QPyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        q_specs=None
    ):
        if q_specs==None:
            return f_linear(input, weight)
        
        wl = q_specs['mantissa_bits']
        blk_rows = q_specs['blk_rows']   # 1
        blk_cols = q_specs['blk_cols']   # B
        rounding = q_specs['rounding']
        block_granularity = q_specs.get('block_granularity', 'fine')
        doing_mx = q_specs.get('doing_mx')
        E = q_specs.get('E')
        M = q_specs.get('M')
        E_bias = q_specs.get('E_bias')
        
        if bias is not None:
            ctx.has_bias = True
            bf_bias = block_quantizer_fine(bias, wl=wl, block_size_rows=1, 
                                         block_size_cols=1, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)
        else:
            ctx.has_bias = False
            bf_bias = None

        # Block dimensions for weight and activations are always transposed         
        qis_input = block_quantizer_fine(input, wl=wl, block_size_rows=blk_rows, 
                                         block_size_cols=blk_cols, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)  # 1XB

        qis_weight = block_quantizer_fine(weight, wl=wl, block_size_rows=blk_rows, 
                                         block_size_cols=blk_cols, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE) # 1XB (linear includes weight transpose - Bx1)
        
        if blk_rows==blk_cols:
#             print('entered')
            ctx.save_for_backward(qis_input, qis_weight)
        else:
            ctx.save_for_backward(input, weight)

        # print(f'forward operands {qis_input.shape, qis_weight.shape}')
        # compute output
        output = f_linear(qis_input, qis_weight)  # 1XB  X   Tr(1XB)
        # print(f'output {output.shape}')
        # print(f'bias {bf_bias.shape}')
        if bias is not None:
            output = output + bf_bias
            
        ctx.q_specs = q_specs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        input, weight = ctx.saved_tensors

        q_specs = ctx.q_specs
        wl = q_specs['mantissa_bits']
        blk_rows = q_specs['blk_rows']
        blk_cols = q_specs['blk_cols']
        rounding = q_specs['rounding']
        block_granularity = q_specs.get('granularity', 'fine')
        doing_mx = q_specs.get('doing_mx')
        E = q_specs.get('E')
        M = q_specs.get('M')
        E_bias = q_specs.get('E_bias')
                
        out_dim = weight.shape[0]
        in_dim = weight.shape[1]


        #####################################################
        # perform tiling operation for grad_weight, grad_bias
        #####################################################
        # Assume inputs are always 2D shapes
        
        if blk_rows==blk_cols:
#             print('entered')
            qex_input = input.to(DEVICE)
            qex_grad_output = block_quantizer_fine(grad_output, wl=wl, block_size_rows=blk_cols, 
                                         block_size_cols=blk_rows, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)
        else:
            qex_input = block_quantizer_fine(input, wl=wl, block_size_rows=blk_cols, 
                                         block_size_cols=blk_rows, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)   # BX1
            qex_grad_output = block_quantizer_fine(grad_output, wl=wl, block_size_rows=blk_cols, 
                                         block_size_cols=blk_rows, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)    # BX1

        
        # compute grad_weight [out_features, in_features]
        qex_grad_output = qex_grad_output.reshape(-1, out_dim)
        qex_input = qex_input.reshape(-1, in_dim)

        # print(f'grad weight operands {qex_grad_output.transpose(0, 1).shape, qex_input.shape}')
        # Compute grad_weight
        grad_weight = torch_matmul(qex_grad_output.transpose(0, 1), qex_input)  # Tr(BX1) X  (BX1)
        # print(f'grad weight {grad_weight.shape}')
        
        want_to_quantize_weight_gradient = 0
        if (want_to_quantize_weight_gradient):
            grad_weight = block_quantizer_fine(grad_weight, wl=wl, block_size_rows=blk_rows, 
                                         block_size_cols=blk_cols, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)
        
        #####################################################
        # perform tiling operation for grad_input
        #####################################################

        if blk_rows==blk_cols:
#             print('entered')
            qos_weight = weight.to(DEVICE)
            qos_grad_output = block_quantizer_fine(grad_output, wl=wl, block_size_rows=blk_rows, 
                                         block_size_cols=blk_cols, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)
        else:
            qos_weight = block_quantizer_fine(weight , wl=wl, block_size_rows=blk_cols, 
                                         block_size_cols=blk_rows, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)   # BX1
            qos_grad_output = block_quantizer_fine(grad_output, wl=wl, block_size_rows=blk_rows, 
                                             block_size_cols=blk_cols, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)  # 1XB
        
        # print(f'grad input operands: {qos_grad_output.shape, qos_weight.shape}')
        # Compute grad_input
        grad_input = torch_matmul(qos_grad_output, qos_weight)   # (1XB)  x  (BX1)
        # print(f'grad input {grad_input.shape}')
        
        

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)
            grad_bias = block_quantizer_fine(grad_bias, wl=wl, block_size_rows=1, 
                                         block_size_cols=1, rounding=rounding, block_granularity=block_granularity, doing_mx=doing_mx, E=E, M=M, E_bias=E_bias).to(DEVICE)
            
        

        return (grad_input, grad_weight, grad_bias, None, None, None)


def custom_qpylinear(
    input,
    weight,
    bias=None,
    q_specs=None
):
    if q_specs is None:
        return f_linear(input, weight, bias=bias)

    return QPyLinearFunction.apply(input, weight, bias, q_specs)


class QPyLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        q_specs=None
    ):
        self.q_specs = q_specs
        super().__init__(in_features, out_features, bias)

    def forward(self, inputs):
        if self.q_specs is None:
            return super().forward(inputs)

        return custom_qpylinear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            q_specs=self.q_specs
        )
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, q_specs={self.q_specs}'



def swish(x):
    return x * torch.sigmoid(x)
