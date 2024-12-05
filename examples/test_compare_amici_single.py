import os
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../")

import time
import yaml

import cProfile
from main_class import *
from matplotlib import cm

# Load the configuration file
amici_config = './system_amici.yml'
single_config = './system_single.yml'

amici_system = HSSystem(config_file_path=amici_config)

disp_amici = amici_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]

single_system = HSSystem(config_file_path=single_config)

disp_single = single_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]

print(disp_single)

# Compare the two systems
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
params = {'axes.labelsize': 90/2.5,'axes.titlesize':90/2.5, 'legend.fontsize': 90/2.5, 'xtick.labelsize': 70/2.5, 'ytick.labelsize': 70/2.5}
matplotlib.rcParams.update(params)
plt.rcParams.update(params)
fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
ax = fig.add_subplot(111)
plt.rcParams.update({'font.size': 90/2.5})
plt.plot(torch.linspace(450, 650, 28), 1000*disp_amici, label='Amici (AP)')
plt.plot(torch.linspace(450, 650, 28), 1000*disp_single, label='Single (SP)')
plt.legend()
ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
ax.set_ylabel('Spreading [µm]', fontsize=90/2.5)
ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
plt.savefig("/home/lpaillet/Documents/Codes/DiffOptics/system_comparison_with_zemax/" + "spreading_curves_amici_single.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()






print(1000*(disp_amici[0] - disp_amici[-1]))
print(1000*(disp_single[0] - disp_single[-1]))

print(single_system.central_positions_wavelengths(torch.Tensor([450., 520., 650.]))[1][:,1])


shift_value = torch.Tensor([-42.850,  0.00329,  38.452])
shift_value -= torch.Tensor([-39.7427,   0.0780,  43.3486])

I = 0
wavelength_id = 0
I = torch.roll(I, shifts=shift_value[wavelength_id].item(), dims=1)
