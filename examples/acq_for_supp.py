import os
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

import sys
sys.path.append("../")
import diffoptics as do

import time
from utils import *
import yaml

import cProfile
from main_class import *
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = [(0, 0, 0), (0.45, 0.45, 0.45), (0.75, 0.75, 0.75), (0.9, 0.9, 0.9), (1, 1, 1)]
custom_gray_cmap = LinearSegmentedColormap.from_list("Custom", colors, N=2000)

systems_files = ["/home/lpaillet/Documents/Codes/Networks/system_amici.yml", "/home/lpaillet/Documents/Codes/Networks/system_amicimis.yml", "/home/lpaillet/Documents/Codes/Networks/system_single.yml","/home/lpaillet/Documents/Codes/Networks/system_singlemis.yml"]

systems = ["amici", "amicimis", "single", "singlemis"]

oversample = 4

mask = torch.load("/home/lpaillet/Documents/Codes/DiffOptics/examples/mask.pt", map_location='cpu')

#texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512].astype('float32')

""" for i in range(1,11):
    texture = scipy.io.loadmat(f"/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/TSA_simu_data/Truth/scene{i:02d}.mat")['img'][:512,:512].astype('float32')
    plt.imshow(texture.sum(-1))
    plt.show() """

texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/TSA_simu_data/Truth/scene07.mat")['img'][:512,:512].astype('float32')


texture = torch.from_numpy(np.transpose(texture, (2,0,1))).float()

batch = texture.unsqueeze(0)

batch = torch.clamp(torch.nn.functional.interpolate(batch, scale_factor=(2, 2), mode='bilinear', align_corners=True), 0, 1)

texture = batch.permute(0, 2, 3, 1) # batchsize x H x W x nC

texture = torch.mul(texture, mask[None, :, :, None])

#texture_oversampled = torch.nn.functional.interpolate(texture, scale_factor=(1, self.oversample), mode='bilinear', align_corners=True)
texture = torch.nn.functional.interpolate(texture, scale_factor=(1, oversample), mode='bilinear', align_corners=True)


for system_file, system in zip(systems_files, systems):
    optics = HSSystem(config_file_path=system_file)

    wavelengths = torch.linspace(450, 650, 112).float()
    optics.wavelengths = wavelengths

    nb_rays = 20

    z0 = torch.tensor([optics.system[-1].d_sensor*torch.cos(optics.system[-1].theta_y*np.pi/180).item() + optics.system[-1].origin[-1] + optics.system[-1].shift[-1]]).item()
    
    big_uv = torch.load(f"rays_{system}.pt", map_location='cpu')
    big_mask = torch.load(f"rays_valid_{system}.pt", map_location='cpu')

    airy = torch.load("airy_amici.pt", map_location='cpu').float()

    batch_acq = optics.render_batch_based_on_saved_pos(big_uv = big_uv, big_mask = big_mask, texture = texture, nb_rays=nb_rays, wavelengths = wavelengths,
                                z0=z0).float()
    del big_uv, big_mask

    batch_acq = F.conv2d(batch_acq.permute(0, 3, 1, 2), airy, padding = airy.shape[-1]//2, groups=optics.wavelengths.shape[0])
    batch_acq = batch_acq.permute(0, 2, 3, 1)

    batch_acq = batch_acq.flip(1)
    batch_acq = batch_acq.sum(-1)[0,:,:]

    plt.figure(figsize=(32, 18), dpi=60)
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.imshow(image.sum(-1), cmap='gray')
    plt.imshow(batch_acq, cmap=custom_gray_cmap)
    plt.savefig("/home/lpaillet/Documents/Codes/DiffOptics/system_comparison_with_zemax/" + f"acquisition_{system}.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.close()
    del batch_acq, airy, optics
