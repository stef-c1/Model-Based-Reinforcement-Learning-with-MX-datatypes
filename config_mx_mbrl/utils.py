import numpy as np
import tensorflow as tf
import torch
from torch import nn as nn


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    tf.compat.v1.disable_eager_execution()
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(config=cfg)
    x = tf.random.truncated_normal(shape=size, stddev=std)
    val = sess.run(x)
    # val = tf.random.truncated_normal(shape=size, stddev=std)
    
    # Close the session and free resources
    sess.close()
    # val = torch.randn(size)
    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b