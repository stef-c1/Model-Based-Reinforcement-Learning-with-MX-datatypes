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

import pickle


from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
import time

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

# def get_layer(model, name):
#     layer = model
#     for attr in name.split("."):
#         layer = getattr(layer, attr)
#     return layer

# def set_layer(model, name, layer):
#     try:
#         attrs, name = name.rsplit(".", 1)
#         model = get_layer(model, attrs)
#     except ValueError:
#         pass
#     setattr(model, name, layer)
    

# def replace_linear_layers(model, quantized=True, q_specs={}):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             linear = get_layer(model, name)
#             # Create new in layer, conv.bias is a tensor or None, bias needs to be bool
#             b = False if linear.bias is None else True
            
#             if quantized:
#                 new_linear = QPyLinear(linear.in_features, linear.out_features, bias=b, q_specs=q_specs)  
#             else:
#                 new_linear = nn.Linear(linear.in_features, linear.out_features, bias=b)  

#             with torch.no_grad():
#                 new_linear.weight.copy_(linear.weight)
#                 if b: 
#                     new_linear.bias.copy_(linear.bias)
#             set_layer(model, name, new_linear)
        

# Get training data


def main3(task='reacher', ind=0, dir='', confs=[(8,8,8),(8,8,8)], epochs=[100,100], batch_size=8, lr=1e-4, dataformats=['FP32',['BSINT8', 'BSE4M3', 'FP32']], split_dataset_percent=0, do_rewind=0, rewind_threshold=5.0, rewind_activation_epoch=20, rewind_mode='reshuffle'):
    weight_lists = {}
    weight_lists['cartpole'] = ['2022-09-20--16-54-24',
                            '2022-09-20--16-48-45',
                            '2022-09-20--17-05-48',
                            '2022-09-20--17-00-08',
                            '2022-09-20--17-11-29',
                            ]

    weight_lists['pusher'] =    ['2022-09-20--17-11-41',
                        '2022-09-20--18-03-12',
                        '2022-09-20--18-55-21',
                        '2022-09-20--19-47-10',
                        '2022-09-20--20-39-33',
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
                
            
            
    # ind=0
    data_path = weight_lists[task][ind]
    dat = loadmat(f'{task}_runs_relu_random/{data_path}/logs.mat')

    # confs = [(8,8,8), (8,1,8)]   # (precision bits, block rows, block columns)
    # confs = [(8,1,32), (8,1,32)]   # (precision bits, block rows, block columns)

    model_list = {}

    # Preprocessing robotics input data to feed into NN
    new_train_in, new_train_targs =[], []
    for obs, acs in zip(dat['observations'][0:], dat['actions'][0:]): #obs is 151x17, acs is 150x7
        new_train_in.append(np.concatenate([obs_preproc(obs[:-1]), acs], axis=-1))
        new_train_targs.append(targ_proc(obs[:-1], obs[1:]))
    train_in2 = np.concatenate(new_train_in, axis=0)
    train_targs2 = np.concatenate(new_train_targs, axis=0)    

    # Splitting the dataset if needed
    if (split_dataset_percent==0):
        idxs = np.random.randint(train_in2.shape[0], size=[train_in2.shape[0]])
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
    else:
        idxs_tot = np.random.randint(train_in2.shape[0], size=[train_in2.shape[0]])
        idxs_cloud = idxs_tot[0:round(split_dataset_percent/100*len(idxs_tot))-1]
        idxs_robot = idxs_tot[round(split_dataset_percent/100*len(idxs_tot)):]
        idxs_lst = [idxs_cloud, idxs_robot]

    # DNN setup
    cfg_model = {}
    cfg_model["acts_relu"] = True
    cfg_model["num_nets"] = 1
    cfg_model["model_pretrained"] = False
    cfg_model['use_adam']  = True
    cfg_model['use_adam_8bit']  = False
    cfg_model['lr'] = lr
    

    # Split of adaptive training configurations
    # epochs = [20, 80]
    # epochs = [80, 20]

    #(mode, E, M, E_bias)
    format_conf_dict = {'FP32':(2,8,23,127), 'BSINT8':(0,0,0,0), 'BSE4M3':(1,4,3,7), 'BSE5M2':(1,5,2,15), 'BSE2M3':(1,2,3,1), 'BSE3M2':(1,3,2,3), 'BSE2M1':(1,2,1,1), 
                        'FP16':(3,5,10,15), 'FP8 (E4M3)':(3,4,3,7), 'FP8 (E5M2)':(3,5,2,15), 'FP15 (E4M10)': (3,4,10,7),
                        'BSE2M1-W/BSE4M3':(1,[2,4,4,4],[1,3,3,3],[1,7,7,7]), 'BSE2M1-WA/BSE4M3':(1,[2,2,4,4],[1,1,3,3],[1,1,7,7]), 
                        'BSE2M3-W/BSE4M3':(1,[2,4,4,4],[3,3,3,3],[1,7,7,7]), 'BSE2M3-WA/BSE4M3':(1,[2,2,4,4],[3,3,3,3],[1,1,7,7]), 
                        'BSE3M2-W/BSE4M3':(1,[3,4,4,4],[2,3,3,3],[3,7,7,7]), 'BSE3M2-WA/BSE4M3':(1,[3,3,4,4],[2,2,3,3],[3,3,7,7]), }

    loss_stores_formats = {}
    loss_stores_formats_tot = {}
    # Main training loop. Outer - Different configurations. Inner - Across different epochs
    
    
    #Make sure that only 1 dataformat for the first part of training
    assert(len(dataformats[0])==1)

    if (rewind_activation_epoch < 15):
        rewind_activation_epoch = 15

    loss_stores = {}
    very_old_model = 0
    average_loss = 0

    for index, configs in enumerate(confs):
        if (split_dataset_percent!=0):
            idxs = idxs_lst[index]
        else:
            assert(len(idxs)==15150)
        for format_ in dataformats[index]:
            print(format_)
            loss_stores["adaptive"] = []
            loss_stores["total"] = []
            cntr = 0
            epoch_conf = epochs[index]//2 if task=="halfcheetah" else epochs[index]
            num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
            cfg_model["q_specs"] = {}
            cfg_model["q_specs"]['mantissa_bits']  = configs[0]
            cfg_model["q_specs"]['blk_rows']  = configs[1]
            cfg_model["q_specs"]['blk_cols']  = configs[2]
            cfg_model["q_specs"]['rounding']  = "nearest"
            
            
            if index==0:
                #format = dataformats[0][0]
                cfg_model["q_specs"]['mode'] = format_conf_dict[format_][0]
                cfg_model["q_specs"]['E'] = format_conf_dict[format_][1]
                cfg_model["q_specs"]['M'] = format_conf_dict[format_][2]
                cfg_model["q_specs"]['E_bias'] = format_conf_dict[format_][3]

                model = module_config.nn_constructor(cfg_model).to(TORCH_DEVICE)
                model.apply(truncated_normal_init)
                model.fit_input_stats(train_in2)

                

            else:
                #format = format_
                cfg_model["q_specs"]['mode'] = format_conf_dict[format_][0]
                cfg_model["q_specs"]['E'] = format_conf_dict[format_][1]
                cfg_model["q_specs"]['M'] = format_conf_dict[format_][2]
                cfg_model["q_specs"]['E_bias'] = format_conf_dict[format_][3]

                model = module_config.nn_constructor(cfg_model).to(TORCH_DEVICE)
                saved_weights = loss_stores[f"weights_adaptive_{index-1}"]
                saved_weights["inputs_mu"]  = torch.nn.Parameter(saved_weights["inputs_mu"].squeeze())
                saved_weights["inputs_sigma"]  = torch.nn.Parameter(saved_weights["inputs_sigma"].squeeze())
                model.load_state_dict(saved_weights, strict=False)

                
            stop_training = 0
            reshuffle_data = 0
            for _ in range(epoch_conf):
                cntr +=1
                batch_ave_loss = []
                if (_ % 10 == 0):
                    print(cntr)

                if (_ % 5 == 0):
                    very_very_old_model = very_old_model #very_very_old_model is 5 to 10 epochs old
                    very_old_model = copy.deepcopy(model)

                
                batch_num = 0
                while (batch_num < (num_batch-1)):
                    batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                    train_in = torch.from_numpy(train_in2[batch_idxs]).to(TORCH_DEVICE).float()
                    train_targ = torch.from_numpy(train_targs2[batch_idxs]).to(TORCH_DEVICE).float()

                    loss = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
                    mean, logvar = model(train_in, ret_logvar=True)
                    inv_var = torch.exp(-logvar)
                    train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                    train_losses = train_losses.mean(-1).sum()
                    loss += train_losses


                    # skip_batch = (loss - loss_stores["adaptive"][-1]) > rewind_threshold
                    if (do_rewind & (cntr > rewind_activation_epoch) & ((loss - average_loss) > rewind_threshold)):
                        model = copy.deepcopy(old_model)
                        if (re_entered == 0):
                            re_entered = 1
                            nb_batches_triggered += 1
                            if (nb_batches_triggered > num_batch*0.2): #trigger of early stopping or reshuffle
                                if (rewind_mode == 'early stopping'):
                                    stop_training = 1
                                elif (rewind_mode == 'reshuffle'):
                                    reshuffle_data = 1
                                else:
                                    raise Exception("Rewind mode is gibberish!")
                                model = very_very_old_model
                                break
                        else:
                            batch_num += 1
                            re_entered = 0


                    else:
                        if (do_rewind & (cntr > rewind_activation_epoch-1)):
                            old_model = copy.deepcopy(model)

                        model.optim.zero_grad()
                        loss.backward()
                        model.optim.step()
                        batch_ave_loss.append(loss.cpu().data/batch_size)

                        re_entered = 0
                        batch_num += 1

                nb_batches_triggered = 0                    

                loss_stores["adaptive"].append(np.mean(batch_ave_loss))
                average_loss = loss_stores["adaptive"][-1]

                if (split_dataset_percent!=0):
                    train_in = torch.from_numpy(train_in2[idxs_tot]).to(TORCH_DEVICE).float()
                    train_targ = torch.from_numpy(train_targs2[idxs_tot]).to(TORCH_DEVICE).float()
                    loss = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
                    mean, logvar = model(train_in, ret_logvar=True)
                    inv_var = torch.exp(-logvar)
                    train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                    train_losses = train_losses.mean(-1).sum()
                    loss += train_losses
                    loss_stores["total"].append(loss.cpu().data/len(idxs_tot))

                if (stop_training):
                    stop_training = 0
                    break
                if (reshuffle_data):
                    reshuffle_data = 0
                    idxs = np.random.shuffle(idxs)


            loss_stores[f"weights_adaptive_{index}"] = copy.deepcopy(model.state_dict())
        
            if index==0:
                loss_stores_formats['cloud'] = loss_stores["adaptive"]
                loss_stores_formats_tot['cloud'] = loss_stores["total"]
            else:
                loss_stores_formats[format_] = loss_stores["adaptive"]
                loss_stores_formats_tot[format_] = loss_stores["total"]


    #GRAPHS
    os.makedirs(dir, exist_ok=True)

    #save loss for later:
    with open(dir+'/loss_dict.pkl', 'wb') as f:
        pickle.dump(loss_stores_formats, f)

    with open(dir+'/loss_dict_tot.pkl', 'wb') as f:
        pickle.dump(loss_stores_formats_tot, f)

    with open(dir+'/inputs.txt', 'w') as f:
        f.write('task='+str(task)+', '+'ind='+str(ind)+', '+'confs='+str(confs)+', '+'epochs='+str(epochs)+', '+'batch_size='+str(batch_size)+', '+'lr='+str(lr)+', '+'dataformats='+str(dataformats))


    colors = {'FP32':[1,0,0,1],'BSINT8':[0,0,1,1],'BSE4M3':[0,1,0,1],'BSE5M2':[0,0.5,0,1],'BSE3M2':[1,1,0,1],'BSE2M3':[0.6,0.6,0,1],'BSE2M1':[1,0.5,0,1],
               'FP16':[0.3,0,0,1], 'FP8 (E4M3)':[14/255, 204/255, 163/255, 1], 'FP8 (E5M2)':[10/255, 69/255, 56/255, 1], 'FP15 (E4M10)': [1,1,1,1],
               'BSE2M1-W/BSE4M3':[110/255, 88/255, 63/255, 1], 'BSE2M1-WA/BSE4M3':[227/255, 180/255, 52/255, 1],
               'BSE2M3-W/BSE4M3':[97/255,79/255,30/255,1], 'BSE2M3-WA/BSE4M3':[1,1,1,1], 
               'BSE3M2-W/BSE4M3':[227/255,208/255,157/255,1], 'BSE3M2-WA/BSE4M3':[1,1,1,1]}
    legend = []
    if (confs[0] == confs[1]):
        if (confs[0][1] == confs[0][2]):
            suffix = ' (S'+str(confs[0][1])+')'
        elif (confs[0][1] == 1):
            suffix = ' (V'+str(confs[0][2])+')'
        else:
            suffix = ' (?)'
    else:
        suffix = ' (adaptive)'

    plt.figure()
    for format_ in dataformats[1]:
        plt.plot([*range(epochs[0]-1, epochs[0]+epochs[1])], [loss_stores_formats['cloud'][-1]]+loss_stores_formats[format_], color=colors[format_])
        if (format_ not in legend):
            if (format_[0]=='F'):
                legend.append(format_)
            else:
                legend.append(format_+suffix)

    plt.legend(legend)

    format_ = dataformats[0][0]
    print(dataformats)
    print('cloud:'+format_)
    plt.plot(loss_stores_formats['cloud'], color=colors[format_])
                
    plt.title('Training Loss for '+str(task)+':'+str(ind))
    plt.xlabel('Training Epochs')
    plt.ylabel('Normalized training loss')
    plt.ylim(top=5)
    plt.savefig(dir+'/fig1.png')


    if (split_dataset_percent!=0):
        legend2 = []
        plt.figure()
        for format_ in dataformats[1]:
            plt.plot([*range(epochs[0]-1, epochs[0]+epochs[1])], [loss_stores_formats_tot['cloud'][-1]]+loss_stores_formats_tot[format_], color=colors[format_])
            if (format_ not in legend2):
                if (format_[0]=='F'):
                    legend2.append(format_)
                else:
                    legend2.append(format_+suffix)

        plt.legend(legend2)

        format_ = dataformats[0][0]
        print(dataformats)
        print('cloud:'+format_)
        plt.plot(loss_stores_formats_tot['cloud'], color=colors[format_])
                    
        plt.title('Total Loss for '+str(task)+':'+str(ind))
        plt.xlabel('Training Epochs')
        plt.ylabel('Normalized total loss')
        plt.ylim(top=5)
        plt.savefig(dir+'/fig2.png')







#runs to do:
tasks = ['reacher', 'cartpole', 'pusher', 'halfcheetah']
cloud_formats = ['FP32', 'FP16', 'FP8 (E4M3)', 'FP8 (E5M2)', 'FP15 (E4M10)']
rewind_thresholds = [2.0,5.0,10.0]
rewind_modes = ['early stopping', 'reshuffle']
epoch = 125

robot_dataformats = ['BSINT8', 'BSE4M3', 'BSE5M2', 'BSE2M3', 'BSE3M2', 'BSE2M1','BSE2M1-W/BSE4M3','BSE2M3-W/BSE4M3','BSE3M2-W/BSE4M3']

for task in tasks:
    for cloud_format in cloud_formats:
        for rewind_threshold in rewind_thresholds:
            for rewind_mode in rewind_modes:
                dir_ = f'auto/{task}/{cloud_format}/rewind_threshold_{str(int(rewind_threshold))}/{rewind_mode}'
                fig_dir = dir_ + '/fig2.png'
                if (os.path.exists(fig_dir)):
                    None
                else:
                    if task=='reacher':
                        ind = 5
                    else:
                        ind = 0

                    if task=='halfcheetah':
                        epochs = [50,50]
                    else:
                        epochs = [epoch, epoch]
                    
                    robot_dataformats_here = copy.deepcopy(robot_dataformats)
                    robot_dataformats_here.append(cloud_format)
                    
                    
                    main3(task=task, ind=ind, dir=dir_, confs=[(8,8,8),(8,8,8)], epochs=epochs, batch_size=8, lr=1e-4, dataformats=[[cloud_format], robot_dataformats_here], 
                          split_dataset_percent=50, do_rewind=1, rewind_threshold=rewind_threshold, rewind_activation_epoch=epochs[0]/5, rewind_mode=rewind_mode)





