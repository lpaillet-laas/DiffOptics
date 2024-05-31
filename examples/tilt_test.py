import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../")
import diffoptics as do

import time
from utils import *

# initialize a lens
device = torch.device('cpu')
save_dir = './render_pattern_demo/'

### Optical Design #############################################################################################################################
R = 12.5
d = 120.0
angle = 53.4/2
H = R*2

tc = 4.1
te = 2
e = tc-te
x0 = R

F = 75

prism_length = 2*H*np.tan(angle*np.pi/180)

offset_collim = 2.4

# Coefs for the surfaces
coefs = torch.tensor([[0, 0, 0, 0, 0, 0],                                              # Start of lens 1
                      [x0+e, 0, 0, -(R+e-x0)/(x0*x0), 0, -(R+e-x0)/(x0*x0)],           # End of lens 1
                      [0, 0, 0, 0, 0, 0],                                              # Prism 1
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],                        # Prism 2
                      [0, 0, -np.tan(angle*np.pi/180), 0, 0, 0],                       # Prism 3
                      [0, 0, 0, 0, 0, 0],                                              # End of prism
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],                           # Ghost surface
                      [x0-e, 0, 0, (R+e-x0)/(x0*x0), 0, (R+e-x0)/(x0*x0)],             # Start of lens 2
                      [0, 0, 0, 0, 0, 0]]).float()                                     # End of lens 2

# Surfaces definition
surfaces1 = [
    do.XYPolynomial(R, F - te, J=1, ai=coefs[0][:3], device = device),                    # Start of lens 1
    do.XYPolynomial(R, F - x0, J=2, ai=coefs[1], device = device),                    # End of lens 1
    do.XYPolynomial(R, 2*F + offset_collim - prism_length/2, J=1, ai=coefs[2][:3], device = device),                                                       # Prism 1
    do.XYPolynomial(R, 2*F + offset_collim - prism_length/2 + R*np.tan(angle*np.pi/180), J=1, ai=coefs[3][:3], device = device),                           # Prism 2
    do.XYPolynomial(R, 2*F + offset_collim - prism_length/2 + (H+R)*np.tan(angle*np.pi/180), J=1, ai=coefs[4][:3], device = device),                       # Prism 3
    do.XYPolynomial(R, 2*F + offset_collim + prism_length/2, J=1, ai=coefs[5][:3], device = device),                         # End of prism
]

surfaces2 = [
    do.XYPolynomial(R, 3*F + 2*offset_collim - x0, J=2, ai=coefs[7], device = device),           # Start of lens 2
    do.XYPolynomial(R, 3*F + 2*offset_collim + te, J=1, ai=coefs[8][:3], device = device),       # End of lens 2
]

# Materials definition
materials1 = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air'),
    do.Material('N-SK2'),
    do.Material('N-SF4'),
    do.Material('N-SK2'),
    do.Material('air'),
]

materials2 = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]

################################################################################################################################################

d_R = 12.7                # 1/2*Height of the optical objects (in mm)
d = 100.0                 # Origin of the optical objects
angle = 53.4/2            # Angle of the prism

d_x1 = 1.767909           # x1 is the distance from the origin (d) to the first prism
d_x2 = 4.531195
d_x3 = 8.651783
x1_e = d_x1
x2_e = 7-d_x2
x3_e = 9.5-d_x3
d_length = 9.5
d_x0 = d_R

d_H = d_R*2
d_prism_length = 2*d_H*np.tan(angle*np.pi/180)

d_F = 75

d = 2*d_F - d_prism_length/2

d_offset_collim = 5.5

d_coefs = torch.tensor([[d_x0 - x1_e, 0, 0, (d_R+x1_e-d_x0)/(d_x0*d_x0), 0, (d_R+x1_e-d_x0)/(d_x0*d_x0)],      # Start of doublet 1
                      [d_x0 + x2_e, 0, 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0)],            # Middle of doublet 1
                      [d_x0 + x3_e, 0, 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0)],            # End of doublet 1
                      [0, 0, 0, 0, 0, 0],                                                          # Prism 1
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],                                    # Prism 2
                      [0, 0, -np.tan(angle*np.pi/180), 0, 0, 0],                                   # Prism 3
                      [0, 0, 0, 0, 0, 0],                                                          # End of prism
                      [d_x0 - x3_e, 0, 0, (d_R+x3_e-d_x0)/(d_x0*d_x0), 0, (d_R+x3_e-d_x0)/(d_x0*d_x0)],              # Start of doublet 2
                      [d_x0 - x2_e, 0, 0, (d_R+x2_e-d_x0)/(d_x0*d_x0), 0, (d_R+x2_e-d_x0)/(d_x0*d_x0)],              # Middle of doublet 2
                      [d_x0 + x1_e, 0, 0, -(d_R+x1_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x1_e-d_x0)/(d_x0*d_x0)]]).float()   # End of doublet 2


d_surfaces1 = [
    do.XYPolynomial(d_R, d_F - d_length/2 + d_x1-d_x0, J=2, ai=d_coefs[0], device = device),              # Start of doublet 1                                                     
    do.XYPolynomial(d_R, d_F - d_length/2 + d_x2-d_x0, J=2, ai=d_coefs[1], device = device),              # Middle of doublet 1                                                     
    do.XYPolynomial(d_R, d_F - d_length/2 + d_x3-d_x0, J=2, ai=d_coefs[2], device = device),              # End of doublet 1                                                     
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2, J=1, ai=d_coefs[3][:3], device = device),                                                                   # Prism 1
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2 + d_R*np.tan(angle*np.pi/180), J=1, ai=d_coefs[4][:3], device = device),                                       # Prism 2
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2 + (d_H+d_R)*np.tan(angle*np.pi/180), J=1, ai=d_coefs[5][:3], device = device),                                   # Prism 3
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim + prism_length/2, J=1, ai=d_coefs[6][:3], device = device),                                     # End of prism                                              
]

d_surfaces1 = [
    do.XYPolynomial(d_R, d_F + d_length/2 - d_x3-d_x0, J=2, ai=d_coefs[7], device = device),              # Start of doublet 1                                                     
    do.XYPolynomial(d_R, d_F + d_length/2 - d_x2-d_x0, J=2, ai=d_coefs[8], device = device),              # Middle of doublet 1                                                     
    do.XYPolynomial(d_R, d_F + d_length/2 - d_x1-d_x0, J=2, ai=d_coefs[9], device = device),              # End of doublet 1                                                     
    #do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2, J=1, ai=d_coefs[3][:3], device = device),                                                                   # Prism 1
    #do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2 + d_R*np.tan(angle*np.pi/180), J=1, ai=d_coefs[4][:3], device = device),                                       # Prism 2
    #do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2 + (d_H+d_R)*np.tan(angle*np.pi/180), J=1, ai=d_coefs[5][:3], device = device),                                   # Prism 3
    #do.XYPolynomial(d_R, 2*d_F + d_offset_collim + prism_length/2, J=1, ai=d_coefs[6][:3], device = device),                                     # End of prism                                              
]


d_materials1 = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('sf5'),
    do.Material('air'),
    #do.Material('N-SK2'),
    #do.Material('N-SF4'),
    #do.Material('N-SK2'),
    #do.Material('air'),
]

d_materials1 = [
    do.Material('air'),
    do.Material('sf5'),
    do.Material('N-BK7'),
    do.Material('air'),
    do.Material('N-SK2'),
    do.Material('N-SF4'),
    do.Material('N-SK2'),
    do.Material('air'),
]

d_surfaces2 = [
    do.XYPolynomial(d_R, 3*d_F + 2*d_offset_collim + d_length/2 - d_x3-d_x0, J=2, ai=d_coefs[7], device = device),  # Start of doublet 2                                                   
    do.XYPolynomial(d_R, 3*d_F + 2*d_offset_collim + d_length/2 - d_x2-d_x0, J=2, ai=d_coefs[8], device = device),  # Middle of doublet 2                                                   
    do.XYPolynomial(d_R, 3*d_F + 2*d_offset_collim + d_length/2 - d_x1-d_x0, J=2, ai=d_coefs[9], device = device),  # End of doublet 2                                                    
]

d_surfaces2 = [
    do.XYPolynomial(d_R, 3*d_F + 2*d_offset_collim - d_length/2 + d_x1-d_x0, J=2, ai=d_coefs[0], device = device),  # Start of doublet 2                                                   
    do.XYPolynomial(d_R, 3*d_F + 2*d_offset_collim - d_length/2 + d_x2-d_x0, J=2, ai=d_coefs[1], device = device),  # Middle of doublet 2                                                   
    do.XYPolynomial(d_R, 3*d_F + 2*d_offset_collim - d_length/2 + d_x3-d_x0, J=2, ai=d_coefs[2], device = device),  # End of doublet 2                                                    
]

""" d_surfaces2 = [
    do.XYPolynomial(d_R, -(2*d_F + d_offset_collim + prism_length/2) + 3*d_F + 2*d_offset_collim - d_length/2 + d_x1-d_x0, J=2, ai=d_coefs[0], device = device),  # Start of doublet 2                                                   
    do.XYPolynomial(d_R, -(2*d_F + d_offset_collim + prism_length/2) + 3*d_F + 2*d_offset_collim - d_length/2 + d_x2-d_x0, J=2, ai=d_coefs[1], device = device),  # Middle of doublet 2                                                   
    do.XYPolynomial(d_R, -(2*d_F + d_offset_collim + prism_length/2) + 3*d_F + 2*d_offset_collim - d_length/2 + d_x3-d_x0, J=2, ai=d_coefs[2], device = device),  # End of doublet 2                                                    
] """


d_materials2 = [
    do.Material('air'),
    do.Material('sf5'),
    do.Material('N-BK7'),
    do.Material('air')
]

d_materials2 = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('sf5'),
    do.Material('air')
]
################################################################################################################################################


tilt_angle = 2.68
tilt_angle = 1.2
tilt_angle = 0
shift_value = -58
shift_value = 0

lens1 = do.Lensgroup(device=device, theta_y=[0], origin=np.array([0,0,0]), shift=np.array([0,0,0]))
last_surface_distance = 2*F + offset_collim + prism_length/2
lens2 = do.Lensgroup(device=device, theta_y=[tilt_angle], origin=np.array([0,0,last_surface_distance]), shift=np.array([shift_value,0,-last_surface_distance]))


lens1.load(surfaces1, materials1)
lens1.d_sensor = 3*F + 2*d_offset_collim # distance to sensor (in mm?)
lens1.r_last = R # radius of sensor (in mm?) in plot

lens1.film_size = [131, 131] # [pixels]
lens1.pixel_size = 80.0e-3*3 # [mm]

lens2.load(surfaces2, materials2)
lens2.d_sensor = 4*F + 2*d_offset_collim # distance to sensor (in mm?)
lens2.r_last = R # radius of sensor (in mm?) in plot

lens2.film_size = [131, 131] # [pixels]
lens2.pixel_size = 80.0e-3*3 # [mm]

################################################################################################################################################

d_tilt_angle = -8.8863
d_tilt_angle = 0 
d_shift_value = -58
d_shift_value = 0

d_lens1 = do.Lensgroup(device=device, theta_y=[0], origin=np.array([0,0,0]), shift=np.array([0,0,0]))
d_last_surface_distance = 2*d_F + d_offset_collim + prism_length/2
#d_last_surface_distance = 0
d_lens2 = do.Lensgroup(device=device, theta_y=[d_tilt_angle],
                       origin=np.array([0,0,d_last_surface_distance]), shift=np.array([d_shift_value,0,-d_last_surface_distance]))


d_lens1.load(d_surfaces1, d_materials1)
d_lens1.d_sensor = 3*d_F + 2*d_offset_collim # distance to sensor (in mm?)
d_lens1.r_last = d_R # radius of sensor (in mm?) in plot

d_lens1.film_size = [131, 131] # [pixels]
d_lens1.pixel_size = 80.0e-3*3 # [mm]

d_lens2.load(d_surfaces2, d_materials2)
d_lens2.d_sensor = 4*d_F + 2*d_offset_collim # distance to sensor (in mm?)
d_lens2.r_last = d_R # radius of sensor (in mm?) in plot

d_lens2.film_size = [131, 131] # [pixels]
d_lens2.pixel_size = 80.0e-3*3 # [mm]

#for surface in d_lens2.surfaces:
#    surface.d += 2*d_F + d_offset_collim + prism_length/2

################################################################################################################################################

if __name__=='__main__':     
    lens_test_setup1 = [{'type': 'XYPolynomial',
                         'params': { 'J':2,
                                    'ai': [d_x0 - x3_e, 0, 0, (d_R+x3_e-d_x0)/(d_x0*d_x0), 0, (d_R+x3_e-d_x0)/(d_x0*d_x0)]
                             },
                         'd': d_F + d_length/2 - d_x3-d_x0,
                         'R': 12.7
                        },
                        {'type': 'XYPolynomial',
                         'params': { 'J':2,
                                    'ai': [d_x0 - x2_e, 0, 0, (d_R+x2_e-d_x0)/(d_x0*d_x0), 0, (d_R+x2_e-d_x0)/(d_x0*d_x0)],
                             },
                         'd': - d_x2 + d_x3,
                         'R': 12.7
                        },
                        {'type': 'XYPolynomial',
                         'params': { 'J':2,
                                    'ai': [d_x0 + x1_e, 0, 0, -(d_R+x1_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x1_e-d_x0)/(d_x0*d_x0)],
                             },
                         'd': - d_x1 + d_x2,
                         'R': 12.7
                        },
                        {'type': 'XYPolynomial',
                         'params': { 'J':1,
                                    'ai': [0,0,0],
                             },
                         'd': d_F + d_offset_collim - d_length/2 - prism_length/2 + d_x1 + d_x0,
                         'R': 12.7
                        },
                        {'type': 'XYPolynomial',
                         'params': { 'J':1,
                                    'ai': [0,0,np.tan(angle*np.pi/180)],
                             },
                         'd': d_R*np.tan(angle*np.pi/180),
                         'R': 12.7
                        },
                        {'type': 'XYPolynomial',
                         'params': { 'J':1,
                                    'ai': [0,0,-np.tan(angle*np.pi/180)],
                             },
                         'd': d_H*np.tan(angle*np.pi/180),
                         'R': 12.7
                        },
                        {'type': 'XYPolynomial',
                         'params': { 'J':1,
                                    'ai': [0,0,0],
                             },
                         'd': d_R*np.tan(angle*np.pi/180),
                         'R': 12.7
                        }]

    lenstest_materials1 = ['air', 'sf5', 'N-BK7', 'air', 'N-SK2', 'N-SF4', 'N-SK2', 'air']

    lens_test_setup2 = [{'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 - x1_e, 0, 0, (d_R+x1_e-d_x0)/(d_x0*d_x0), 0, (d_R+x1_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': 3*d_F + 2*d_offset_collim - d_length/2 + d_x1-d_x0,
                 'R': 12.7
                 },
                 {'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 + x2_e, 0, 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': d_x2 - d_x1,
                 'R': 12.7
                 },
                 {'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 + x3_e, 0, 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': d_x3 - d_x2,
                 'R': 12.7
                 }]
    
    lens_test_setup3 = [{'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 - x1_e, 0, 0, (d_R+x1_e-d_x0)/(d_x0*d_x0), 0, (d_R+x1_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': 4*d_F + 3*d_offset_collim - d_length/2 + d_x1-d_x0,
                 'R': 12.7
                 },
                 {'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 + x2_e, 0, 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': d_x2 - d_x1,
                 'R': 12.7
                 },
                 {'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 + x3_e, 0, 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': d_x3 - d_x2,
                 'R': 12.7
                 }]
    
    lens_test_setup4 = [{'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 - x1_e, 0, 0, (d_R+x1_e-d_x0)/(d_x0*d_x0), 0, (d_R+x1_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': 5*d_F + 4*d_offset_collim - d_length/2 + d_x1-d_x0,
                 'R': 12.7
                 },
                 {'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 + x2_e, 0, 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': d_x2 - d_x1,
                 'R': 12.7
                 },
                 {'type': 'XYPolynomial',
                 'params': {
                        'J': 2,
                        'ai': [d_x0 + x3_e, 0, 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0)],
                    },
                 'd': d_x3 - d_x2,
                 'R': 12.7
                 }]

    lenstest_materials2 = ['air', 'N-BK7', 'sf5', 'air']
    print(d_F - d_length/2)
    print(2*d_F+5.5-d_prism_length/2 - (d_F + d_length/2))
    print(3*d_F - d_length/2 + 2*5.5 - (2*d_F + 5.5 + d_prism_length/2))
    print(4*d_F + 2*5.5 - (3*d_F + d_length/2 + 2*5.5))
    print(d_prism_length)
    lens_test1 = create_lensgroup(lens_test_setup1, lenstest_materials1,
                                 d_sensor=3*d_F + 2*d_offset_collim, r_last=d_R,
                                 film_size=[131,131], pixel_size=80.0e-3*3, device=device)
    
    last_surface_distance = 2*d_F + d_offset_collim + d_prism_length/2
    d_tilt_angle = -8.8863

    lens_test2 = create_lensgroup(lens_test_setup2, lenstest_materials2,
                                 d_sensor=4*d_F + 2*d_offset_collim, r_last=d_R,
                                 film_size=[131,131], pixel_size=80.0e-3*3,
                                 theta_y=[d_tilt_angle],
                                 origin=np.array([0,0,d_last_surface_distance]),
                                 shift=np.array([d_shift_value,0,-d_last_surface_distance]), device=device)
    
    lens_test3 = create_lensgroup(lens_test_setup3, lenstest_materials2,
                                 d_sensor=5*d_F + 3*d_offset_collim, r_last=d_R,
                                 film_size=[131,131], pixel_size=80.0e-3*3,
                                 theta_y=[-d_tilt_angle],
                                 origin=np.array([0,0,d_last_surface_distance]),
                                 shift=np.array([-45,0,-d_last_surface_distance]), device=device)
    
    lens_test4 = create_lensgroup(lens_test_setup4, lenstest_materials2,
                                 d_sensor=6*d_F + 4*d_offset_collim, r_last=d_R,
                                 film_size=[131,131], pixel_size=80.0e-3*3,
                                 theta_y=[0],
                                 origin=np.array([0,0,d_last_surface_distance]),
                                 shift=np.array([-15,0,-d_last_surface_distance]), device=device)
    
    lenses = [d_lens1, d_lens2]
    lens_test1.pixel_size = 80.0e-3
    lens_test2.pixel_size = 80.0e-3
    #lenses = [lens_test1, lens_test2]
    #lenses = [lens_test1, lens_test2, lens_test3, lens_test4]

    #plot_setup_basic_rays(lenses)
    
    #lens_test2.theta_y = -lens_test2.theta_y
    #lens_test2.update()
    #wavelengths = np.linspace(450, 550, 55)
    #wavelengths = [656.2725, 587.5618, 486.1327]
    #propagate(lenses, wavelengths = wavelengths, nb_rays = 1, z0 = 4*d_F + 11, offsets = [-d_x0, 0], save_dir = save_dir)

    print(plot_spot_diagram(lenses, 550, 13, 80*1e-3*130, save_dir = save_dir))

    N = 10000
    max_angle = 5
    x_pos, y_pos, z_pos = 0, 0, 0.1
    x_pos, y_pos, z_pos = 0, 0, 0.1
    angles_phi = 2*max_angle*np.random.rand(N) - max_angle
    angles_psi = 2*max_angle*np.random.rand(N) - max_angle
    angles = np.stack((angles_phi, angles_psi), axis=-1)
    wavelength = 550.0

    ps = trace_psf_from_point_source(lenses = lenses, angles = angles, x_pos = x_pos, y_pos = y_pos, z_pos = z_pos, wavelength = wavelength,
                      normalize = False, show = False)

    plt.hist2d(ps[...,0], ps[...,1], bins=100)
    plt.show()