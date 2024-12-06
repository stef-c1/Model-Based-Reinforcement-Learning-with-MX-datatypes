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

import json
import ast
from pathlib import Path
import pickle
import time

from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
import torch.nn.functional as F

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


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer

def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)




        

def main_funct(task='reacher', ind=5, batch_size=32, lr=1e-3, dataformats={'BSE4M3':(4,3,7),'BSINT8':(0,0,0),'FP32':(0,0,0)}, dirpath='', weight_grouping=0):
    # Get training data
    weight_lists = {}
    weight_lists['cartpole'] = ['2022-09-20--16-54-24',
                            '2022-09-20--16-48-45',
                            '2022-09-20--17-05-48',
                            '2022-09-20--17-00-08',
                            '2022-09-20--17-11-29'
                            ]

    weight_lists['pusher'] =    ['2022-09-20--17-11-41',
                        '2022-09-20--18-03-12',
                        '2022-09-20--18-55-21',
                        '2022-09-20--19-47-10',
                        '2022-09-20--20-39-33'
                        ]

    weight_lists['reacher'] =   ['2022-09-21--02-59-39',
                            '2022-09-20--22-04-39',
                            '2022-09-21--00-30-23',
                            '2022-09-20--17-09-13',
                            '2022-09-20--19-38-38',
                            'logs'
                            ]

    weight_lists['halfcheetah'] =  [ '2022-09-21--16-52-24',
                                '2022-09-21--16-53-08',
                                '2022-09-23--09-46-22',
                                '2022-09-24--02-55-44',
                                '2022-09-24--05-29-44'
                                ]




    TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # task = "cartpole"
    # task = "pusher"
    # task = "reacher" 
    # task = "halfcheetah" 
    # ind=1

    # batch_size = 32


    # dataformats = {'BSE4M3':(4,3,7), 'BSE5M2':(5,2,15), 'BSE2M3':(2,3,1), 'BSE3M2':(3,2,3), 'BSE2M1':(2,1,1), 'BSINT8':(0,0,0), 'FP32':(0,0,0)}
    # dataformats = {'BSE4M3':(4,3,7),'BSINT8':(0,0,0),'FP32':(0,0,0)}
    # dataformats = {'BSFP8':((4,3,7),(5,2,15)),'BSINT8':(0,0,0),'FP32':(0,0,0)}
    # dataformats = {'FP32':(0,0,0)}
    # dataformats = {'BSINT8':(0,0,0)}
    # dataformats = {'BSE4M3':(4,3,7)}

    module_config = reacher_config()
            
    if task == "cartpole": 
        module_config = cartpole_config()
    elif task == "pusher": 
        module_config = pusher_config()
    elif task == "reacher":
        module_config = reacher_config()
    elif task == "halfcheetah":
        module_config = halfcheetah_config()
                
            
    if hasattr(module_config, "UPDATE_FNS"):
        update_fns = module_config.UPDATE_FNS
    else:
        update_fns = lambda obs: obs
    if hasattr(module_config, "obs_preproc"):
        obs_preproc = module_config.obs_preproc
    else:
        obs_preproc = lambda obs: obs
    if hasattr(module_config, "obs_postproc"):
        obs_postproc = module_config.obs_postproc
    else:
        obs_postproc = lambda obs, model_out: model_out
    if hasattr(module_config, "targ_proc"):
        targ_proc = module_config.targ_proc
    else:
        targ_proc = lambda obs, next_obs: next_obs
                
            
            

    data_path = weight_lists[task][ind]
    dat = loadmat(f'{task}_runs_relu_random/{data_path}/logs.mat')

    # confs = [(8,8,8), (8,1,8)]   # (precision bits, block rows, block columns)
    confs = [(8,1,32), (8,1,32)]   # (precision bits, block rows, block columns) 
    if (confs[0]==confs[1]):
        if (confs[0][1]==1):
            blocksize = 'V'+str(confs[0][2])
        elif (confs[0][1]==confs[0][2]):
            blocksize = 'S'+str(confs[0][1])
        else:
            blocksize = 'ill defined'
    else:
        blocksize = 'adaptive' #...




    epochs = [10,10] #[30, 120]

    ###ON CLOUD######################################################################
    # Preprocessing robotics input data to feed into NN
    new_train_in, new_train_targs =[], []
    for obs, acs in zip(dat['observations'][0:], dat['actions'][0:]):
        new_train_in.append(np.concatenate([obs_preproc(obs[:-1]), acs], axis=-1))
        new_train_targs.append(targ_proc(obs[:-1], obs[1:]))
    train_in2 = np.concatenate(new_train_in, axis=0)
    train_targs2 = np.concatenate(new_train_targs, axis=0)    

    idxs = np.random.randint(train_in2.shape[0], size=[train_in2.shape[0]])
    num_batch = int(np.ceil(idxs.shape[-1] / batch_size))


    #Splitting dataset in cloud and field
    #loss curves should be on full dataset, but only trained on part


    loss_stores = {}
    loss_stores["adaptive"] = []
    loss_per_batch = []
    logvar_per_batch = []
    max_logvar_per_batch = []
    min_logvar_per_batch = []

    # DNN setup
    cfg_model = {}
    cfg_model["acts_relu"] = True
    cfg_model["num_nets"] = 1
    cfg_model["model_pretrained"] = False
    cfg_model['use_adam']  = True #True
    cfg_model['use_adam_8bit']  = False
    cfg_model['lr'] = lr
    cntr = 0

    configs = confs[0]
    epoch_conf = epochs[0]
    num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
    cfg_model["q_specs"] = {}
    cfg_model["q_specs"]['mantissa_bits']  = configs[0]
    cfg_model["q_specs"]['blk_rows']  = configs[1]
    cfg_model["q_specs"]['blk_cols']  = configs[2]
    cfg_model["q_specs"]['rounding']  = "nearest"
    cfg_model["q_specs"]['doing_mx'] = 2
    cfg_model["q_specs"]['E'] = 0
    cfg_model["q_specs"]['M'] = 0
    cfg_model["q_specs"]['E_bias'] = 0

    model = module_config.nn_constructor(cfg_model).to(TORCH_DEVICE)
    model.apply(truncated_normal_init)
    model.fit_input_stats(train_in2)


    for _ in range(epoch_conf):
        cntr +=1
        batch_ave_loss = []
        print(cntr)
        for batch_num in range(num_batch-1):
            batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
            train_in = torch.from_numpy(train_in2[batch_idxs]).to(TORCH_DEVICE).float()
            train_targ = torch.from_numpy(train_targs2[batch_idxs]).to(TORCH_DEVICE).float()

            loss = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
            max_logvar_save = copy.deepcopy(model.max_logvar)
            min_logvar_save = copy.deepcopy(model.min_logvar)
            mean, logvar = model(train_in, ret_logvar=True)
            logvar_save = copy.deepcopy(logvar.detach())
            inv_var = torch.exp(-logvar)
            train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
            train_losses = train_losses.mean(-1).sum()
            loss += train_losses
            model.optim.zero_grad()
            loss.backward()
            model.optim.step()
            batch_ave_loss.append(loss.cpu().data/batch_size)
            loss_per_batch.append(loss.cpu().data/batch_size)
            max_logvar_per_batch.append(max_logvar_save.cpu().data)
            min_logvar_per_batch.append(min_logvar_save.cpu().data)
            logvar_per_batch.append(logvar_save.cpu().data)


        loss_stores["adaptive"].append(np.mean(batch_ave_loss))
    loss_stores[f"weights_adaptive_{0}"] = copy.deepcopy(model.state_dict())

    indices_dict = None

    model_copy2 = copy.deepcopy(model)

    ###Weight GROUPING###############################################################
    if weight_grouping:
        model_copy = copy.deepcopy(model.state_dict())
        print(model_copy.keys())
        weight_keys_to_sort = ['fc1.weight', 'fc2.weight', 'fc3.weight', 'fc4.weight']
        indices_dict = {}
        for key in weight_keys_to_sort:
            layer = model_copy[key]

            #process (a copy of) the weights to shape that they will be quantized in
            x = layer
            block_size_rows = confs[0][1]
            block_size_cols = confs[0][2]
            block_granularity = 'fine' #I choose default because this is seemingly never set
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

            nh, remainder = np.divmod(x.size(2), kh)
            nh += bool(remainder)
            nv, remainder =  np.divmod(x.size(1), kv)
            nv += bool(remainder)

            ph, pv = nh*kh - x.size(2), nv*kv - x.size(1)

            x_pad = F.pad(x, (0, ph, 0, pv, 0, 0))
            x_pad_unfold = x_pad.unfold(1,kv,sv).unfold(2,kh,sh).reshape(-1, kv, kh).float()


            #sort them, save indices as key to go back and forth between sorted and unsorted
            x_flat = x_pad_unfold.flatten()
            x_sort, indices = x_flat.sort()
            indices_dict[key] = indices

    ###ON ROBOT######################################################################
    loss_dict_robot = {}
    loss_per_batch_robot_dict = {}
    max_logvar_per_batch_robot_dict = {}
    min_logvar_per_batch_robot_dict = {}
    logvar_per_batch_robot_dict = {}
    for format_ in dataformats:
        cfg_model = {}
        cfg_model["acts_relu"] = True
        cfg_model["num_nets"] = 1
        cfg_model["model_pretrained"] = False
        cfg_model['use_adam']  = True #True
        cfg_model['use_adam_8bit']  = False
        cfg_model['lr'] = lr

        configs = confs[1]
        cfg_model["q_specs"] = {}
        cfg_model["q_specs"]['mantissa_bits']  = configs[0]
        cfg_model["q_specs"]['blk_rows']  = configs[1]
        cfg_model["q_specs"]['blk_cols']  = configs[2]
        cfg_model["q_specs"]['rounding']  = "nearest"

        model = module_config.nn_constructor(cfg_model).to(TORCH_DEVICE)
        model.apply(truncated_normal_init)
        model.fit_input_stats(train_in2)


        if (format_ == 'BSINT8'):
            doing_mx = 0 #0 is BFP, 1 is MX, 2 is no quantization (FP32) 
        elif (format_ == 'FP32'):
            doing_mx = 2
            print('doing_mx='+str(doing_mx))
        else:
            doing_mx = 1
            
        EMB_tuple = dataformats[format_]
        E = EMB_tuple[0]
        M = EMB_tuple[1]
        E_bias = EMB_tuple[2]

        model.q_specs['doing_mx'] = doing_mx
        model.q_specs['E'] = E
        model.q_specs['M'] = M
        model.q_specs['E_bias'] = E_bias



        print(get_layer(model_copy2,'fc1'))

        layer1 = copy.deepcopy(get_layer(model_copy2,'fc1'))
        layer2 = copy.deepcopy(get_layer(model_copy2,'fc2'))
        layer3 = copy.deepcopy(get_layer(model_copy2,'fc3'))
        layer4 = copy.deepcopy(get_layer(model_copy2,'fc4'))

        layer1.q_specs['doing_mx'] = doing_mx
        layer1.q_specs['E'] = E
        layer1.q_specs['M'] = M
        layer1.q_specs['E_bias'] = E_bias

        layer2.q_specs['doing_mx'] = doing_mx
        layer2.q_specs['E'] = E
        layer2.q_specs['M'] = M
        layer2.q_specs['E_bias'] = E_bias

        layer3.q_specs['doing_mx'] = doing_mx
        layer3.q_specs['E'] = E
        layer3.q_specs['M'] = M
        layer3.q_specs['E_bias'] = E_bias
        
        layer4.q_specs['doing_mx'] = doing_mx
        layer4.q_specs['E'] = E
        layer4.q_specs['M'] = M
        layer4.q_specs['E_bias'] = E_bias

        set_layer(model,'fc1',layer1)
        set_layer(model,'fc2',layer2)
        set_layer(model,'fc3',layer3)
        set_layer(model,'fc4',layer4)


            # # Main training loop. Outer - Different configurations. Inner - Across different epochs
            # cfg_model["q_specs"] = {}
            # cfg_model["q_specs"]['mantissa_bits']  = configs[0]
            # cfg_model["q_specs"]['blk_rows']  = configs[1]
            # cfg_model["q_specs"]['blk_cols']  = configs[2]
            # cfg_model["q_specs"]['rounding']  = "nearest"
            # cfg_model["q_specs"]['doing_mx'] = doing_mx
            # cfg_model["q_specs"]['E'] = E
            # cfg_model["q_specs"]['M'] = M
            # cfg_model["q_specs"]['E_bias'] = E_bias
            # cfg_model["q_specs"]['indices_dict'] = indices_dict
            
            # model = module_config.nn_constructor(cfg_model).to(TORCH_DEVICE)

            # # Getting saved model
            # saved_weights = loss_stores[f"weights_adaptive_{0}"]
            # saved_weights["inputs_mu"]  = torch.nn.Parameter(saved_weights["inputs_mu"].squeeze())
            # saved_weights["inputs_sigma"]  = torch.nn.Parameter(saved_weights["inputs_sigma"].squeeze())
            # model.load_state_dict(saved_weights, strict=False)


        loss_stores_robot = {}
        loss_stores_robot["adaptive"] = []

        cntr = epochs[0]

        configs = confs[1]
        epoch_conf = epochs[1]//2 if task=="halfcheetah" else epochs[1]
        
        loss_per_batch_robot = []
        max_logvar_per_batch_robot = []
        min_logvar_per_batch_robot = []
        logvar_per_batch_robot = []
        for _ in range(epoch_conf):
            cntr +=1
            batch_ave_loss = []
            print(cntr)
            for batch_num in range(num_batch-1):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                train_in = torch.from_numpy(train_in2[batch_idxs]).to(TORCH_DEVICE).float()
                train_targ = torch.from_numpy(train_targs2[batch_idxs]).to(TORCH_DEVICE).float()

                loss = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
                max_logvar_save = copy.deepcopy(model.max_logvar)
                min_logvar_save = copy.deepcopy(model.min_logvar)
                mean, logvar = model(train_in, ret_logvar=True)
                logvar_save = copy.deepcopy(logvar.detach())
                inv_var = torch.exp(-logvar)
                train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                train_losses = train_losses.mean(-1).sum()
                loss += train_losses
                model.optim.zero_grad()
                loss.backward()
                model.optim.step()
                batch_ave_loss.append(loss.cpu().data/batch_size)
                loss_per_batch_robot.append(loss.cpu().data/batch_size)
                max_logvar_per_batch_robot.append(max_logvar_save.cpu().data)
                min_logvar_per_batch_robot.append(min_logvar_save.cpu().data)
                logvar_per_batch_robot.append(logvar_save.cpu().data)

            loss_stores_robot["adaptive"].append(np.mean(batch_ave_loss))


        loss_dict_robot[format_] = loss_stores_robot["adaptive"]
        loss_per_batch_robot_dict[format_] = loss_per_batch_robot
        max_logvar_per_batch_robot_dict[format_] = max_logvar_per_batch_robot
        min_logvar_per_batch_robot_dict[format_] = min_logvar_per_batch_robot
        logvar_per_batch_robot_dict[format_] = logvar_per_batch_robot



    #PLOTTING ######################

    path = Path(dirpath+'loss')
    path.mkdir(parents=True, exist_ok=True)
    file_name =  path / ('loss_json_v32_id' + str(ind))
    file_name2 = path / ('loss_json_v32_id_batch' + str(ind))
    file_name3a = path / ('max_lgvr_json_batch' +str(ind))
    file_name3b = path / ('min_lgvr_json_batch' +str(ind))
    file_name4 = path / ('logvar_json_batch' +str(ind))

    comp_results = loss_dict_robot
    comp_results['cloud'] = loss_stores["adaptive"]
    keys_with_results = comp_results.keys()

    loss_per_batch_robot_dict['cloud'] = loss_per_batch
    max_logvar_per_batch_robot_dict['cloud'] = max_logvar_per_batch
    min_logvar_per_batch_robot_dict['cloud'] = min_logvar_per_batch
    logvar_per_batch_robot_dict['cloud'] = logvar_per_batch

    # add_to_existing = 0
    # if (add_to_existing):
    #     #read old file
    #     file = open(file_name, 'r')
    #     comp_results_read = json.load(file)
    #     file.close()
    #     comp_results = ast.literal_eval(comp_results_read)
    #     #add new part
    #     comp_results.update(loss_dict_robot)

    write_back = 1
    if (write_back):
        file = open(file_name, 'w+')
        json.dump(str(comp_results), file)
        file.close()
        file2 = open(file_name2, 'w+')
        json.dump(str(loss_per_batch_robot_dict), file2)
        file2.close()
        file3a = open(file_name3a, 'w+')
        json.dump(str(max_logvar_per_batch_robot_dict), file3a)
        file3a.close()
        file3b = open(file_name3b, 'w+')
        json.dump(str(min_logvar_per_batch_robot_dict), file3b)
        file3b.close()
        file4 = open(file_name4, 'w+')
        json.dump(str(logvar_per_batch_robot_dict), file4)
        file2.close()


    #Graphs
    suffix = ' ('+blocksize+')'
    keys = ['cloud', 'BSE2M1', 'BSE3M2', 'BSE2M3', 'BSE5M2', 'BSE4M3', 'BSINT8', 'FP32', 'BSFP8']
    legend_complete = ['BSE2M1'+suffix, 'BSE3M2'+suffix, 'BSE2M3'+suffix, 'BSE5M2'+suffix, 'BSE4M3'+suffix, 'BSINT8'+suffix, 'FP32', 'BSFP8'+suffix]
    legend = []

    color_dict = {'FP32':[1,0,0,1],'BSINT8':[0,0,1,1],'BSE4M3':[0,1,0,1],'BSE5M2':[0,0.5,0,1],'BSE3M2':[1,1,0,1],'BSE2M3':[0.6,0.6,0,1],'BSE2M1':[1,0.5,0,1],'BSFP8':[0,0.75,0,1]}

    plt.figure()
    for key in keys:
        if (key in keys_with_results):
            if (key == 'cloud'):
                None
            else:
                color = color_dict[key]
                plt.plot(comp_results['cloud']+comp_results[key], color=color)
                if (key[0]=='F'):
                    legend.append(key)
                else:
                    legend.append(key+suffix)


 
    plt.legend(legend)
    plt.xlabel('training epochs')
    plt.ylabel('training loss')
    plt.title(task +' training runs (id:'+str(ind)+')')
    ylim_dict = {'reacher':(-400,0), 'cartpole':(-300,0), 'pusher':(-400,0), 'halfcheetah':(-150,200)}
    # plt.ylim(ylim_dict[task][0], ylim_dict[task][1])
    plt.ylim(top=ylim_dict[task][1])
    plt.savefig(dirpath+'fig1.png')



    plt.figure()
    for key in keys:
        if (key in keys_with_results):
            if (key == 'cloud'):
                None
            else:
                color = color_dict[key]
                plt.plot(loss_per_batch_robot_dict['cloud']+loss_per_batch_robot_dict[key], color=color)
                if (key[0]=='F'):
                    legend.append(key)
                else:
                    legend.append(key+suffix)


    plt.legend(legend)
    plt.xlabel('training batches')
    plt.ylabel('training loss')
    plt.title(task +' training runs (id:'+str(ind)+')')
    ylim_dict = {'reacher':(-400,0), 'cartpole':(-300,0), 'pusher':(-400,0), 'halfcheetah':(-150,200)}
    # plt.ylim(ylim_dict[task][0], ylim_dict[task][1])
    plt.ylim(top=ylim_dict[task][1])
    plt.savefig(dirpath+'fig2.png')



# main_funct(task='reacher', ind=5 , batch_size=64, lr=1e-3, dataformats={'FP32':(0,0,0)}, dirpath='trials/run0/')
# main_funct(task='reacher', ind=5 , batch_size=64, lr=1e-3, dataformats={'BSE4M3':(4,3,7), 'BSE5M2':(5,2,15), 'BSE2M3':(2,3,1), 'BSE3M2':(3,2,3), 'BSE2M1':(2,1,1), 'BSINT8':(0,0,0), 'FP32':(0,0,0)}, dirpath='trials/run1/')
# main_funct(task='halfcheetah', ind=0 , batch_size=32, lr=1e-3, dataformats={'BSE4M3':(4,3,7), 'BSINT8':(0,0,0), 'FP32':(0,0,0)}, dirpath='trials/run2/')
# main_funct(task='reacher', ind=5 , batch_size=32, lr=1e-4, dataformats={'BSE4M3':(4,3,7), 'BSE5M2':(5,2,15), 'BSE2M3':(2,3,1), 'BSE3M2':(3,2,3), 'BSE2M1':(2,1,1), 'BSINT8':(0,0,0), 'FP32':(0,0,0)}, dirpath='trials/run3/')
# main_funct(task='reacher', ind=5 , batch_size=64, lr=2.5e-3, dataformats={'BSE4M3':(4,3,7), 'BSE5M2':(5,2,15), 'BSE2M3':(2,3,1), 'BSE3M2':(3,2,3), 'BSE2M1':(2,1,1), 'BSINT8':(0,0,0), 'FP32':(0,0,0)}, dirpath='trials/run4/')
# main_funct(task='reacher', ind=5 , batch_size=64, lr=1e-4, dataformats={'BSE4M3':(4,3,7), 'BSE5M2':(5,2,15), 'BSE2M3':(2,3,1), 'BSE3M2':(3,2,3), 'BSE2M1':(2,1,1), 'BSINT8':(0,0,0), 'FP32':(0,0,0)}, dirpath='trials/run5/')


# main_funct(task='halfcheetah', ind=0 , batch_size=32, lr=1e-3, dataformats={'BSINT8':(0,0,0), 'FP32':(0,0,0)}, dirpath='trials/run2/')
# main_funct(task='halfcheetah', ind=0 , batch_size=32, lr=1e-3, dataformats={'BSE4M3':(4,3,7)}, dirpath='trials/run3/')

main_funct(task='reacher', ind=5 , batch_size=64, lr=5e-4, dataformats={'BSE4M3':(4,3,7), 'FP32':(0,0,0), 'BSINT8':(0,0,0)}, dirpath='trials/c_r_test54/', weight_grouping=0)