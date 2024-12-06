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
        self.fc4 = QPyLinear(200, out_features, bias=True, q_specs=self.q_specs)

        self.inputs_mu = nn.Parameter(torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):
        lin0_decays = 0.00025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.0005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.0005 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.00075 * (self.lin3_w ** 2).sum() / 2.0
        return lin0_decays + lin1_decays + lin2_decays + lin3_decays


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

        mean = inputs[:, :self.out_features // 2]

        logvar = inputs[:, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar
        
        return mean, torch.exp(logvar)



class PusherConfigModule:
    ENV_NAME = "MBRLPusher-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 27, 20
    GP_NINDUCING_POINTS = 200

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

        # Keep track of the previous goal pos
        # to determine if we should replace the goal pos on GPU
        # to minimize communication overhead
        self.prev_ac_goal_pos = None
        self.goal_pos_gpu = None

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn(self, obs):
        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], self.ENV.ac_goal_pos

        assert isinstance(obs, torch.Tensor)

        should_replace = False

        # If there was a previous goal pos
        # and the current goal pos is different from the previous goal post...
        if self.prev_ac_goal_pos is not None and (self.prev_ac_goal_pos == goal_pos).all() is False:
            # then we replace the goal pos on GPU
            should_replace = True

        # else if there is no current goal pos...
        elif self.goal_pos_gpu is None:
            # then we also move the goal pos to GPU
            should_replace = True

        if should_replace:
            self.goal_pos_gpu = torch.from_numpy(goal_pos).float().to(TORCH_DEVICE)
            self.prev_ac_goal_pos = goal_pos

        tip_obj_dist = (tip_pos - obj_pos).abs().sum(dim=1)
        obj_goal_dist = (self.goal_pos_gpu - obj_pos).abs().sum(dim=1)

        return to_w * tip_obj_dist + og_w * obj_goal_dist

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
                        PusherConfigModule.MODEL_IN, PusherConfigModule.MODEL_OUT * 2,
                          acts_relu = acts_relu, q_specs=q_specs).to(TORCH_DEVICE)

        if use_adam:
            model.optim = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            model.optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        return model


CONFIG_MODULE = PusherConfigModule
