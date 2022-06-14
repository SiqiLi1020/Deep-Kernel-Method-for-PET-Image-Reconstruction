# import libs
from __future__ import print_function
import matplotlib.pyplot as plt
#% matplotlib inline

import os
import cv
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *
from scipy import io

import torch
import torch.optim
from PIL import Image
from utils.denoising_utils import *
from Networks.Unet2D_softmax_W import UNet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.


# load neighbor index
matr1 = io.loadmat(r'training data/index_300_20e6_n_20.mat')
index = matr1['N']
index = index - 1
index = index.astype(np.int32)
N = torch.from_numpy(index).type(dtype)
N_number = N.size(1)


# Setup
INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
LR = 1e-3
OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 10
exp_weight = 0.99
num_iter = 501
input_depth = 3
figsize = 5

s1 = 111
s2 = 111
s3 = 4
inter_channels = 16
out_channels = 16
net = UNet(4, inter_channels, out_channels)
net = net.type(dtype)

# load normalized composite prior images (4 channels)
prior = open(r'training data/CIP_20e6_n_20.img','rb')
img_noisy_np1 = np.fromfile(prior, dtype=np.float32).reshape((s3,s1,s2))
img_noisy_np1 = img_noisy_np1 / img_noisy_np1.max()
prior_input = np.zeros((s3,s1+1,s2+1))
prior_input[:,0:111,0:111] = img_noisy_np1
img_noisy_np1 = torch.from_numpy(prior_input)
img_noisy_np1 = img_noisy_np1.numpy()
net_input = np_to_torch(img_noisy_np1).type(dtype)

# load noise image obtained from reconstruction on down-sampled projection data
noise = open(r'training data/low_20_20e6_20.img','rb')
noise_img = np.fromfile(noise, dtype=np.float32).reshape((s3,s1,s2))
noise_img = torch.from_numpy(noise_img)
noise_img = noise_img.numpy()
noise_img = np_to_torch(noise_img).type(dtype)
#load prior images as label
target = open(r'training data/CIP_20e6_20.img','rb')
target = np.fromfile(target, dtype=np.float32).reshape((s3,s1,s2))
target = torch.from_numpy(target)
target = target.numpy()
target = np_to_torch(target).type(dtype)
# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0


def closure():
    loss = []
    global i, out_avg, psrn_noisy_last, last_net, net_input, noise_img, target

    result = net(net_input_saved, noise_img, N)
    W = result[0]
    out = result[1]

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = torch.mean(mse(out, target))
    total_loss.requires_grad_(True)
    total_loss.backward()
    loss.append(total_loss.item())
    if PLOT and i % show_every == 0:
        print('Iteration %05d   Loss %f ' % (i, total_loss.item()))
        f = os.path.join(r'your training', 'Unet_{}iter.ckpt'.format(i))
        torch.save(net.state_dict(), f)
        W = W.detach().cpu().numpy()
        io.savemat(r'your training\W_{}iter.mat'.format(i), mdict={'W'.format(i): W})
    i += 1

    return total_loss


p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


