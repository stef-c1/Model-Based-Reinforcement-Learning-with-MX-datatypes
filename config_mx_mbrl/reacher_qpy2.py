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
        
        if acts_relu:
            self.activation = F.relu
        else:
            self.activation = swish

        self.in_features = in_features
        self.out_features = out_features
        self.q_specs = q_specs


        self.fc1 = QPyLinear(in_features, 200, bias=True, q_specs=self.q_specs)
        self.fc2 = QPyLinear(200, 200, bias=True, q_specs=self.q_specs)
        self.fc3 = QPyLinear(200, 200, bias=True, q_specs=self.q_specs)
        self.fc4 = QPyLinear(200, out_features, bias=True, q_specs=self.q_specs)

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

        mean = inputs[:, :self.out_features // 2]

        logvar = inputs[:, self.out_features // 2:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        
        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)



class ReacherConfigModule:
    ENV_NAME = "MBRLReacher3D-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 24, 17
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        # self.ENV = gym.make(self.ENV_NAME)
        # self.ENV.reset()
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }
        self.UPDATE_FNS = [self.update_goal]

        self.goal = None

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def update_goal(self):
        self.goal = self.ENV.goal

    def obs_cost_fn(self, obs):

        assert isinstance(obs, torch.Tensor)
        assert self.goal is not None

        obs = obs.detach().cpu().numpy()

        ee_pos = ReacherConfigModule.get_ee_pos(obs)
        dis = ee_pos - self.goal

        cost = np.sum(np.square(dis), axis=1)

        return torch.from_numpy(cost).float().to(TORCH_DEVICE)

    @staticmethod
    def ac_cost_fn(acs):
        return 0.01 * (acs ** 2).sum(dim=1)

    def nn_constructor(self, model_init_cfg={}):
        ensemble_size = model_init_cfg.get("num_nets", 1)
        acts_relu = model_init_cfg.get("acts_relu", True)
        use_adam = model_init_cfg.get("use_adam", True)
        q_specs =  model_init_cfg.get("q_specs", None)
        lr = model_init_cfg.get("lr", 1e-3)

       # model_init_cfg = {}
        ensemble_size=1  # Set for single models for now 
        model = PtModel(ensemble_size,
                        ReacherConfigModule.MODEL_IN, ReacherConfigModule.MODEL_OUT * 2,
                          acts_relu = acts_relu, q_specs=q_specs).to(TORCH_DEVICE)

        if use_adam:
            model.optim = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            model.optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        return model

    @staticmethod
    def get_ee_pos(states):

        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]

        rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                  axis=1)

        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end


CONFIG_MODULE = ReacherConfigModule
