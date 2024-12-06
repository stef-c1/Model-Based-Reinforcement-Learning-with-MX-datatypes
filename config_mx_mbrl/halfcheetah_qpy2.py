from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
# import gym

import torch
from torch import nn as nn
from torch.nn import functional as F
from QPyLinear import QPyLinear


# from config.utils import swish, get_affine_params


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def swish(x):
    return x * torch.sigmoid(x)

def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1
) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(
            mean, std, size=(bound_violations,), device=tensor.device
        )
    return tensor


class PtModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features, acts_relu = False, q_specs=None):
        super().__init__()

        self.num_nets = ensemble_size

        self.in_features = in_features
        self.out_features = out_features
        self.q_specs = q_specs

        if acts_relu:
            self.activation = F.relu
        else:
            self.activation = swish


        self.fc1 = QPyLinear(in_features, 200, bias=True, q_specs=self.q_specs)
        self.fc2 = QPyLinear(200, 200, bias=True, q_specs=self.q_specs)
        self.fc3 = QPyLinear(200, 200, bias=True, q_specs=self.q_specs)
        self.fc4 = QPyLinear(200, 200, bias=True, q_specs=self.q_specs)
        self.fc5 = QPyLinear(200, out_features, bias=True, q_specs=self.q_specs)

        self.inputs_mu = nn.Parameter(torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)


    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma
        inputs = self.fc1(inputs)
        inputs = self.activation(inputs)    
                
        inputs = self.fc2(inputs)
        inputs = self.activation(inputs)    
        inputs = self.fc3(inputs)
        inputs = self.activation(inputs)    
        inputs = self.fc4(inputs)     
        inputs = self.activation(inputs)    

        inputs = self.fc5(inputs)     

        mean = inputs[ :, :self.out_features // 2]

        logvar = inputs[ :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)


class HalfCheetahConfigModule:
    ENV_NAME = "MBRLHalfCheetah-v0"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 400
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 30
    MODEL_IN, MODEL_OUT = 24, 18
    GP_NINDUCING_POINTS = 300

    def __init__(self):
        # self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, 1:2],
                obs[:, 2:3].sin(),
                obs[:, 2:3].cos(),
                obs[:, 3:]
            ], dim=1)

    @staticmethod
    def obs_postproc(obs, pred):

        assert isinstance(obs, torch.Tensor)

        return torch.cat([
            pred[:, :1],
            obs[:, 1:] + pred[:, 1:]
        ], dim=1)

    @staticmethod
    def targ_proc(obs, next_obs):

        if isinstance(obs, np.ndarray):
            return np.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                next_obs[:, :1],
                next_obs[:, 1:] - obs[:, 1:]
            ], dim=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, 0]

    @staticmethod
    def ac_cost_fn(acs):
        return 0.1 * (acs ** 2).sum(dim=1)

    def nn_constructor(self, model_init_cfg={}):
        ensemble_size = model_init_cfg.get("num_nets", 1)
        acts_relu = model_init_cfg.get("acts_relu", True)
        use_adam = model_init_cfg.get("use_adam", True)
        q_specs =  model_init_cfg.get("q_specs", None)

        ensemble_size=1  # Set for single models for now 
        model = PtModel(ensemble_size,
                        HalfCheetahConfigModule.MODEL_IN, HalfCheetahConfigModule.MODEL_OUT * 2,
                            acts_relu = acts_relu, q_specs=q_specs).to(TORCH_DEVICE)

        if use_adam:
            model.optim = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            model.optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        return model


CONFIG_MODULE = HalfCheetahConfigModule
