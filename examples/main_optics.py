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
import yaml

# initialize a lens
device = torch.device('cpu')
save_dir = './render_pattern_demo/'

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

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

lens_setup1 = [{'type': 'XYPolynomial',
                        'params': { 'J':2,
                                'ai': [d_x0 - x3_e, 0, 0, (d_R+x3_e-d_x0)/(d_x0*d_x0), 0, (d_R+x3_e-d_x0)/(d_x0*d_x0)]
                            },
                    #    'd': d_F + d_length/2 - d_x3-d_x0,
                        'd': 70.120 - d_x0 + x3_e,
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':2,
                                'ai': [d_x0 - x2_e, 0, 0, (d_R+x2_e-d_x0)/(d_x0*d_x0), 0, (d_R+x2_e-d_x0)/(d_x0*d_x0)],
                            },
                        'd': - d_x2 + d_x3,
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':2,
                                'ai': [d_x0 + x1_e, 0, 0, -(d_R+x1_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x1_e-d_x0)/(d_x0*d_x0)],
                            },
                        'd': - d_x1 + d_x2,
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,0],
                            },
                    #    'd': d_F + d_offset_collim - d_length/2 - d_prism_length/2 + d_x1 + d_x0,
                        'd': 75.0 + d_x0 + x1_e,
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,np.tan(angle*np.pi/180)],
                            },
                        'd': d_R*np.tan(angle*np.pi/180),
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,-np.tan(angle*np.pi/180)],
                            },
                        'd': d_H*np.tan(angle*np.pi/180),
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,0],
                            },
                        'd': d_R*np.tan(angle*np.pi/180),
                        'R': d_R
                    }]

lenstest_materials1 = ['air', 'sf5', 'N-BK7', 'air', 'N-SK2', 'N-SF4', 'N-SK2', 'air']

d_tilt_angle = -8.8863  # if middle w is 550nm
#d_tilt_angle = - 9.0969 # if middle w is 520nm
d_tilt_angle = - 9.088 # if middle w is 520nm
d_shift_value = 1.1

lens_setup2 = [{'type': 'XYPolynomial',
                'params': {
                    'J': 2,
                    'ai': [d_x0 - x1_e, 0, 0, (d_R+x1_e-d_x0)/(d_x0*d_x0), 0, (d_R+x1_e-d_x0)/(d_x0*d_x0)],
                },
                #'d': 3*d_F + 2*d_offset_collim - d_length/2 + d_x1-d_x0,
                #'d': 70.120 + 9.5 + 75.0 + d_prism_length + 1/np.cos(d_tilt_angle*np.pi/180)*75 - d_x0 + x1_e,
                'd': 70.120 + 9.5 + 75.0 + d_prism_length + 75.0 - d_x0 + x1_e,
                'R': d_R
                },
                {'type': 'XYPolynomial',
                'params': {
                    'J': 2,
                    'ai': [d_x0 + x2_e, 0, 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x2_e-d_x0)/(d_x0*d_x0)],
                },
                'd': d_x2 - d_x1,
                'R': d_R
                },
                {'type': 'XYPolynomial',
                'params': {
                    'J': 2,
                    'ai': [d_x0 + x3_e, 0, 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0), 0, -(d_R+x3_e-d_x0)/(d_x0*d_x0)],
                },
                'd': d_x3 - d_x2,
                'R': d_R
                }]

lens_test_setup = [{'type': 'XYPolynomial',
                'params': {
                    'J': 2,
                    'ai': [d_x0 - x1_e, 0, 0, (d_R+x1_e-d_x0)/(d_x0*d_x0), 0, (d_R+x1_e-d_x0)/(d_x0*d_x0)],
                },
                #'d': 3*d_F + 2*d_offset_collim - d_length/2 + d_x1-d_x0,
                'd': 154.024 - d_x0 + x1_e,
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


def draw_circle(N, z, max_angle, center_x=0, center_y=0):
    theta = np.random.uniform(0, 2*np.pi, N)
    radius = np.random.uniform(0, (z*np.tan(max_angle*np.pi/180))**2, N)**0.5

    x = radius*np.cos(theta) + center_x
    y = radius*np.sin(theta) + center_y

    return x, y

def draw_circle_hexapolar(N, z, max_angle, center_x=0, center_y=0):
    #theta = np.linspace(0, 2*np.pi, N_theta)
    radius = np.linspace(0, z*np.tan(max_angle*np.pi/180), N)
    x = []
    y = []
    for i, r in enumerate(radius):
        if i==0:
            x.append(0)
            y.append(0)
        else:
            theta = np.linspace(0, 2*np.pi, 6*i, endpoint=False)
            x += list(r*np.cos(theta))
            y += list(r*np.sin(theta))
    x = np.array(x) + center_x
    y = np.array(y) + center_y

    return x, y

def compute_centroid(points):
    points = points if type(points) is torch.Tensor else torch.from_numpy(points)
    return torch.mean(points, dim=0)

def find_bins(points, size_bin, nb_bins=11):
    #min_x, min_y = torch.min(points, dim=0).values
    #max_x, max_y = torch.max(points, dim=0).values

    #bins = torch.zeros((nb_bins, nb_bins))
    bins_i = torch.zeros((nb_bins+1))
    bins_j = torch.zeros((nb_bins+1))
    centroid = compute_centroid(points)

    for i in range(nb_bins+1):
        bins_i[i] = centroid[0] - size_bin/2 - (nb_bins//2)*size_bin + i*size_bin
        bins_j[i] = centroid[1] - size_bin/2 - (nb_bins//2)*size_bin + i*size_bin

    return bins_i, bins_j


if __name__ == '__main__':
    adjustment = 4

    usecase = 'determine_psf'
    
    lens_group1 = create_lensgroup(lens_setup1, lenstest_materials1,
                                 #d_sensor=3*d_F + 2*d_offset_collim,
                                 #d_sensor = 70.120 + d_x3 - d_x1 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + x3_e + x1_e + 75.0 - d_x0 + x1_e,
                                 d_sensor = 70.120 + 9.5 + 75.0 + d_prism_length + 75.0,
                                 r_last=d_R,
                                 film_size=[adjustment*100,adjustment*100], pixel_size=1/adjustment*80.0e-3, device=device)
    
    #d_last_surface_distance = 2*d_F + d_offset_collim + d_prism_length/2
    d_last_surface_distance = 70.120 + d_x3 - d_x1 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + x3_e + x1_e

    lens_group2 = create_lensgroup(lens_setup2, lenstest_materials2,
                                 #d_sensor=4*d_F + 2*d_offset_collim,
                                 #d_sensor = 70.120 + d_x3 - d_x1 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + 2*x3_e + 2*x1_e + 75.0 + d_x3 - d_x1 + 70.120,
                                 d_sensor = 70.120 + 9.5 + 75.0 + d_prism_length + 75.0 + 9.5 + 70.120,
                                 r_last=d_R,
                                 film_size=[adjustment*100,adjustment*100], pixel_size=1/adjustment*80.0e-3,
                                 theta_y=d_tilt_angle,
                                 origin=np.array([0,0,d_last_surface_distance]),
                                 shift=np.array([-d_shift_value,0,-d_last_surface_distance]), device=device)
    
    lens_group_sanity = create_lensgroup(lens_test_setup, lenstest_materials2,
                                         d_sensor = 154.024 + 9.5 + 137.779,
                                         r_last = 4.558, device=device)
    
    """ print(f"Length between object and first lens: {d_F - d_length/2}mm")
    print(f"Length between first lens and prism: {2*d_F+5.5-d_prism_length/2 - (d_F + d_length/2)}mm")
    print(f"Length between prism and second lens: {3*d_F - d_length/2 + 2*5.5 - (2*d_F + 5.5 + d_prism_length/2)}mm")
    print(f"Length between second lens and sensor: {4*d_F + 2*5.5 - (3*d_F + d_length/2 + 2*5.5)}mm")
    print(f"Length of prism: {d_prism_length}mm") # Length of prism """

    if usecase in ['psf', 'spot']:
        #lens_group1.d_sensor = 70.120 + d_x3 - d_x1 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + x3_e + x1_e + 0.5
        #lens_group2.d_sensor = 70.120 + d_x3 - d_x1 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + 2*x3_e + 2*x1_e + 75.0 + d_x3 - d_x1 + 70.120
        lens_group1.d_sensor = 70.120 + 9.5 + 75.0 + d_prism_length + 0.5
        lens_group2.d_sensor = 70.120 + 9.5 + 75.0 + d_prism_length  + 75.0 + 9.5 + 70.120 - 0*1 #+ 10  #- 1.2
        #lens_group2.d_sensor = 1/np.cos(d_tilt_angle*np.pi/180)*(70.120 + 9.5 + 75.0 + 2*d_H*np.tan(angle*np.pi/180))  + 75.0 + 9.5 + 70.120
        #lens_group2.d_sensor = 70.120 + 9.5 + 75.0 + 2*d_H*np.tan(angle*np.pi/180)  + 75.0 + 9.5 + 70.120
    elif usecase == 'render':
        #lens_group1.d_sensor = 70.120 + d_x3 - d_x1 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + x3_e + x1_e + 75.0 - d_x0 + x1_e
        #lens_group2.d_sensor = 70.120 + d_x3 - d_x1 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + 2*x3_e + 2*x1_e + 75.0 + d_x3 - d_x1 + 0.5
        
        #lens_group1.d_sensor = 70.120 + 9.5 + 75.0 + 2*d_H*np.tan(angle*np.pi/180) + np.cos(d_tilt_angle*np.pi/180)*75.0
        #lens_group2.d_sensor = 70.120 + 9.5 + 75.0 + 2*d_H*np.tan(angle*np.pi/180)  + np.cos(d_tilt_angle*np.pi/180)*75.0 + 9.5 + 0.5 + 70.120 - 0.5
        print("hey")

    lenses = [lens_group1, lens_group2]
    #lenses = [lens_group_sanity]
    #plot_setup_basic_rays(lenses)
    for material in lenses[0].materials:
        for wavelength in [450., 550., 650.]:
            print(f"Index {material.name} at {wavelength}nm: {material.ior(wavelength)}")

    if usecase == 'spot':
        sys.path.append('/home/lpaillet/Documents/Codes/simca/')
        from simca.CassiSystem_lightning import CassiSystemOptim
        wavelengths = [450., 550., 650.]
        for w in wavelengths:
            plt.figure()
            ps = plot_spot_diagram(lenses, w, 400, 400, 20*1e-3, save_dir = save_dir, show=False)
            print(ps[...,0].shape)
            print(ps[...,0][::20].shape)
            ps = ps[19::20,:]
            new_ps = np.zeros((ps.shape[0]//20, 2))
            for i in range(new_ps.shape[0]//20):
                new_ps[i*20:(i+1)*20,:] = np.flip(ps[400*i:400*i+20,:], axis=0)
            ps = new_ps
            ps[..., 0] = - ps[..., 0]
            ps[..., 0] -= 0.014
            np.save(save_dir + f'grid_do_{w}.npy', ps)
            plt.scatter(ps[...,0], ps[...,1], color='b')
        
            config = "/home/lpaillet/Documents/Codes/simca/simca/configs/cassi_system_optim_optics_full_triplet_sd_cassi_prism_propag.yml"
            config_system = ""
            with open(config, "r") as file:
                config_system = yaml.safe_load(file)
            cassi_system = CassiSystemOptim(system_config=config_system)

            prop_x, prop_y = cassi_system.propagate_coded_aperture_grid()
            if w == 450.:
                i = 0
            elif w == 550.:
                i=1
            elif w == 650.:
                i=2
            prop_x_flat = prop_x[0,::20,::20,i].detach().numpy().flatten()/1000
            prop_y_flat = prop_y[0,::20,::20,i].detach().numpy().flatten()/1000
            prop_flat = np.stack((prop_x_flat, prop_y_flat), axis=-1)
            np.save(save_dir + f'grid_simca_{w}.npy', prop_flat)
            plt.scatter(prop_x_flat, prop_y_flat, color='r')
            plt.title('Spot diagram at ' + str(w) + 'nm')
            plt.xlabel('x [mm]')
            plt.ylabel('y [mm]')
            plt.legend(['dO', 'SIMCA'])
            print(f"Average distance at {str(w)} nm: {np.mean(np.linalg.norm(ps - prop_flat, axis=-1)):.4f}mm")
            #print(ps)
            #print("break")
            #print(np.stack((prop_x_flat, prop_y_flat), axis=-1))
        plt.show()
    if usecase == 'render':
        lens_group2.theta_y = -lens_group2.theta_y
        lens_group2.update()
        #wavelengths = np.linspace(450, 550, 55)
        wavelengths = [656.2725, 587.5618, 486.1327]
        wavelengths = [450., 550., 650.]
        size_pattern = tuple(lenses[-1].film_size)
        texture = np.kron([[1, 0] * (lenses[-1].film_size[0]//40), [0, 1] * (lenses[-1].film_size[1]//40)] * (lenses[-1].film_size[0]//40), np.ones((20, 20)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 20x20 on a 400x400 size

        propagate(lenses, wavelengths = wavelengths, nb_rays = 10, z0 = 70.120 + 9.5 + 75.0 + 2*d_H*np.tan(angle*np.pi/180)  + np.cos(d_tilt_angle*np.pi/180)*75.0 + 9.5 + 70.120 - 1,
                  texture = texture, offsets = [-d_x0, 0], save_dir = save_dir, aperture_reduction=4)

    """ N = 100
    max_angle = 2
    line_pos, col_pos, z_pos = 0, 0, 0.
    #angles_phi = 2*max_angle*np.random.rand(N) - max_angle
    angles_phi = 2*max_angle*np.linspace(0,1, N) - max_angle
    #angles_psi = 2*max_angle*np.random.rand(N) - max_angle
    angles_psi = 2*max_angle*np.linspace(0,1, N) - max_angle
    angles = np.stack((angles_phi, angles_psi), axis=-1)
    angles = cartesian_product(angles_phi, angles_psi) """
    

    """ N = 1000
    max_angle = 2
    z = 10
    x, y = draw_circle(N, z, max_angle, center_x = line_pos, center_y = col_pos)
    x, y = draw_circle(N, z, max_angle, center_x = 0, center_y = 0)
    d = np.stack((x, y, z*np.ones(N)), axis=-1) #- np.array([line_pos, col_pos, z_pos])
    wavelength = 520.0 """

    if usecase == 'psf':
        N = 100
        N = 58
        max_angle = 22.5
        line_pos, col_pos, z_pos = 0, 0, 0.

        source_pos = np.array([col_pos, line_pos])
        z = max(np.linalg.norm(source_pos - np.array([-12.7, 0])), np.linalg.norm(source_pos - np.array([12.7, 0])), np.linalg.norm(source_pos - np.array([0, -12.7])), np.linalg.norm(source_pos - np.array([0, 12.7])))
        #x, y = draw_circle(N, z, max_angle, center_x = line_pos, center_y = col_pos)
        #x, y = draw_circle(N, z, max_angle, center_x = 0, center_y = 0)

        x, y = draw_circle_hexapolar(N, z, max_angle, center_x = 0, center_y = 0)

        d = np.stack((x, y, 70.120*np.ones(x.shape[0])), axis=-1) #- np.array([line_pos, col_pos, z_pos])
        #d = np.stack((x, y, 154.024*np.ones(N)), axis=-1)
        wavelength = 520.0
        #wavelength = np.linspace(450, 650, 55)[28]

        ps = trace_psf_from_point_source(lenses = lenses, angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = z_pos, wavelength = wavelength,
                        normalize = False, show_rays = True, save_dir=save_dir, d = d, ignore_invalid = False, show_res = True)

        print(ps)
        fig, ax = plt.subplots()
        hist = torch.histogramdd(ps.flip(1), bins=11, density=False)
        hist, edges = hist.hist.numpy(), hist.bin_edges
        #ax.imshow(hist.T, origin='lower', interpolation='nearest', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
        ax.hist2d(ps[...,1], ps[...,0], bins=11)
        ax.axis('equal')
        plt.show()
    

    if usecase == 'psf_database':
        N = 10000
        N = 58
        max_angle = 22.5
        z_pos = 0.

        database_size = 4000
        line_pos_list = np.random.uniform(-2.56, 2.56, size=database_size)
        col_pos_list = np.random.uniform(-2.56, 2.56, size=database_size)
        wavelength_list = np.random.randint(0, 54, size=database_size)

        lenses[-1].d_sensor = lenses[-1].d_sensor - 0.42  #TODO Attention à ça

        for i in range(database_size):
            line_pos = line_pos_list[i]
            col_pos = col_pos_list[i]
            wavelength = np.linspace(450, 650, 55)[wavelength_list[i]]
            source_pos = np.array([col_pos, line_pos])
            z = max(np.linalg.norm(source_pos - np.array([-12.7, 0])), np.linalg.norm(source_pos - np.array([12.7, 0])), np.linalg.norm(source_pos - np.array([0, -12.7])), np.linalg.norm(source_pos - np.array([0, 12.7])))
            #x, y = draw_circle(N, z, max_angle, center_x = line_pos, center_y = col_pos)
            #x, y = draw_circle(N, z, max_angle, center_x = 0, center_y = 0)

            x, y = draw_circle_hexapolar(N, z, max_angle, center_x = 0, center_y = 0)
            d = np.stack((x, y, 70.120*np.ones(x.shape[0])), axis=-1) #- np.array([line_pos, col_pos, z_pos])
            #d = np.stack((x, y, 154.024*np.ones(N)), axis=-1)

            ps = trace_psf_from_point_source(lenses = lenses, angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = z_pos, wavelength = wavelength,
                            normalize = False, show_rays = False, save_dir=save_dir, d = d, ignore_invalid = False, show_res = False)
            
            np.save(save_dir + "/psfs/2.56_database_for_comparison/" f'x_{-col_pos}_y_{-line_pos}_w_{wavelength_list[i]}.npy', ps)
            if i % 100==0:
                print(f"{i} samples generated")

    if usecase == 'psf_field':
        N = 10000
        N = 58
        nb_ray = 1 + 3*N*(N-1)
        max_angle = 22.5

        z_pos = 0.

        lim = 2.7
        nb_pts = 20
        min_w = 450
        max_w = 650
        n_w = 55
        line_pos_list = np.linspace(-lim, lim, nb_pts)
        col_pos_list = np.linspace(-lim, lim, nb_pts)
        wavelength_list = np.linspace(min_w, max_w, n_w)
        #wavelength_list = [520.]

        lenses[-1].d_sensor = lenses[-1].d_sensor - 0.42  #TODO Attention à ça

        time_start = time.time()
        for wavelength in wavelength_list:
            saved_data = np.empty((nb_pts, nb_pts, nb_ray, 2))
            for i, col_pos in enumerate(col_pos_list):
                for j, line_pos in enumerate(line_pos_list):
                    source_pos = np.array([col_pos, line_pos])
                    z = max(np.linalg.norm(source_pos - np.array([-12.7, 0])), np.linalg.norm(source_pos - np.array([12.7, 0])), np.linalg.norm(source_pos - np.array([0, -12.7])), np.linalg.norm(source_pos - np.array([0, 12.7])))
                    #x, y = draw_circle(N, z, max_angle, center_x = line_pos, center_y = col_pos)
                    #x, y = draw_circle(N, z, max_angle, center_x = 0, center_y = 0)
                    x, y = draw_circle_hexapolar(N, z, max_angle, center_x = 0, center_y = 0)
                    d = np.stack((x, y, 70.120*np.ones(x.shape[0])), axis=-1) #- np.array([line_pos, col_pos, z_pos])
                    #d = np.stack((x, y, 154.024*np.ones(N)), axis=-1)
                    

                    ps = trace_psf_from_point_source(lenses = lenses, angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = z_pos, wavelength = wavelength,
                                    normalize = False, show_rays = False, save_dir=save_dir, d = d, ignore_invalid = False, show_res = False)
            
                    saved_data[i, j, :ps.shape[0], :] = ps
                print(f"Line {i} done after {time.time() - time_start:.1f} seconds")
            print(f"Wavelength {wavelength} done after {time.time() - time_start:.1f} seconds")


            np.save(save_dir +'/psfs/' + f'psf_field_w_{round(wavelength, 2)}_lim_{lim}_pts_{nb_pts}.npy', saved_data)
    
    if usecase == 'psf_mid_field':
        N = 10000
        N = 58
        max_angle = 22.5

        z_pos = 0.

        lim = 2.7
        nb_pts = 20
        min_w = 450
        max_w = 650
        n_w = 55
        line_pos_list = np.linspace(-lim, lim, nb_pts)[:-1]
        col_pos_list = np.linspace(-lim, lim, nb_pts)[:-1]
        #line_pos_list = np.linspace(-lim, lim, nb_pts)
        #col_pos_list = np.linspace(-lim, lim, nb_pts)
        wavelength_list = np.linspace(450, 650, 55)
        wavelength_list = [wavelength_list[0], wavelength_list[19], wavelength_list[-1]] # 450, 520.37, 650
        

        lenses[-1].d_sensor = lenses[-1].d_sensor - 0.42  #TODO Attention à ça

        time_start = time.time()
        for wav, wavelength in enumerate(wavelength_list):
            if wav==0:
                w = 0
            elif wav==1:
                w = 19
            elif wav==2:
                w = 54
            for i, col_pos in enumerate(col_pos_list):
                for j, line_pos in enumerate(line_pos_list):
                    new_col_pos = col_pos + 2*lim/(nb_pts-1)/2
                    new_line_pos = line_pos + 2*lim/(nb_pts-1)/2
                    #new_col_pos = col_pos
                    #new_line_pos = line_pos
                    source_pos = np.array([new_col_pos, new_line_pos])
                    z = max(np.linalg.norm(source_pos - np.array([-12.7, 0])), np.linalg.norm(source_pos - np.array([12.7, 0])), np.linalg.norm(source_pos - np.array([0, -12.7])), np.linalg.norm(source_pos - np.array([0, 12.7])))
                    x, y = draw_circle_hexapolar(N, z, max_angle, center_x = 0, center_y = 0)

                    #x, y = draw_circle(N, z, max_angle, center_x = 0, center_y = 0)
                    d = np.stack((x, y, 70.120*np.ones(x.shape[0])), axis=-1) #- np.array([line_pos, col_pos, z_pos])
                    #d = np.stack((x, y, 154.024*np.ones(N)), axis=-1)
                    

                    ps = trace_psf_from_point_source(lenses = lenses, angles = None, x_pos = new_line_pos, y_pos = new_col_pos, z_pos = z_pos, wavelength = wavelength,
                                    normalize = False, show_rays = False, save_dir=save_dir, d = d, ignore_invalid = False, show_res = False)
                    
                    np.save(save_dir + "/psfs/2.56_exact_regular_database_for_comparison/" f'x_{-new_col_pos}_y_{-new_line_pos}_w_{w}.npy', ps)
                print(f"Line {i} done after {time.time() - time_start:.1f} seconds")
                
            print(f"Wavelength {wavelength} done after {time.time() - time_start:.1f} seconds")
    
    if usecase == "determine_psf":
        N = 10000
        N = 58
        nb_ray = 1 + 3*N*(N-1)
        #max_angle = 45.
        max_angle = 22.5
        wavelengths_list = [450.]
        wavelengths_list = [450., 520., 650.]

        base_value = lenses[1].d_sensor
        depth_list = np.arange(-0.5, 0.51, 0.05)
        depth_list = [0.705]
        depth_list = np.arange(-0.47, -0.39, 0.01)
        depth_list = [-0.42]
        #depth_list = np.arange(-1, 0.05, 0.05)
        
        line_pos_list = np.linspace(2.5, -2.5, 7)
        col_pos_list = np.linspace(2.5, -2.5, 7)

        #line_pos_list = [2.5]
        #col_pos_list = [2.5]
        saved_data = np.zeros((len(depth_list), len(wavelengths_list), len(line_pos_list),len(col_pos_list),nb_ray,2))
        z_pos = 0

        for d, depth in enumerate(depth_list):
            lenses[1].d_sensor = base_value + depth
            for w, wavelength in enumerate(wavelengths_list):
                for l, line_pos in enumerate(line_pos_list):
                    for c, col_pos in enumerate(col_pos_list):
                        source_pos = np.array([col_pos, line_pos])
                        z = max(np.linalg.norm(source_pos - np.array([-12.7, 0])), np.linalg.norm(source_pos - np.array([12.7, 0])), np.linalg.norm(source_pos - np.array([0, -12.7])), np.linalg.norm(source_pos - np.array([0, 12.7])))
                        x, y = draw_circle_hexapolar(N, z, max_angle, center_x = 0, center_y = 0)
                        #x, y = draw_circle(N, z, max_angle, center_x = 0, center_y = 0)
                        dist = np.stack((x, y, 70.120*np.ones(x.shape[0])), axis=-1) #- np.array([line_pos, col_pos, z_pos])

                        ps = trace_psf_from_point_source(lenses = lenses, angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = z_pos, wavelength = wavelength,
                                        normalize = False, show_rays = False, save_dir=save_dir, d = dist, ignore_invalid = False, show_res = False)
                        saved_data[d, w, l, c, :ps.shape[0], :] = ps
            print(f"Depth {depth:.2f} done")

        plt.close()
        """ plt.figure()
        plt.scatter(saved_data[0, 1, 1, :, 1], saved_data[0, 1, 1, :, 0])
        plt.figure()
        plt.scatter(saved_data[-1, 1, 1, :, 1], saved_data[-1, 1, 1, :, 0]) """

        """ plt.figure()
        for i in range(3):
            for j in range(3):
                ps = saved_data[0, i, j, :, :]
                ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                plt.scatter(ps[:, 1], ps[:, 0])
        plt.axis('equal') """

        """ for de, depth in enumerate(depth_list):
            fig, ax = plt.subplots(3,3)
            pixel_size = 0.01
            kernel_size = 21
            for i in range(3):
                for j in range(3):
                    ps = saved_data[0, i, j, :, :]
                    ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                    bins_i, bins_j = find_bins(ps, pixel_size, kernel_size)
                    hist = torch.histogramdd(torch.from_numpy(ps).float().flip(1), bins=(bins_j, bins_i), density=False).hist
                    #hist = torch.histogramdd(torch.from_numpy(ps).float().flip(1), bins=11, density=False).hist
                    hist = hist.T.numpy()
                    ax[i,j].imshow(hist.reshape(kernel_size, kernel_size), origin='lower', interpolation='nearest') """
        for w, wavelength in enumerate(wavelengths_list):
            for de, depth in enumerate(depth_list):
                plt.figure()
                for i in range(len(line_pos_list)):
                    for j in range(len(col_pos_list)):
                        ps = saved_data[de, w, i, j, :, :]
                        ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                        plt.scatter(ps[:, 1], ps[:, 0], s=0.1)
                plt.axis('equal')
                plt.title(f"Wavelength {wavelength:.2f}, depth {depth:.2f}")
                plt.savefig(save_dir + '/psfs_optim/' + f"w_{wavelength:.2f}_depth_{depth:.2f}.png", format='png')

                #plt.close()

        plt.show()
        for w, wavelength in enumerate(wavelengths_list):
            for de, depth in enumerate(depth_list):
                fig, ax = plt.subplots(len(line_pos_list),len(col_pos_list))
                pixel_size = 0.005
                kernel_size = 11
                for i in range(len(line_pos_list)):
                    for j in range(len(col_pos_list)):
                        ps = saved_data[de, w, i, j, :, :]
                        ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                        bins_i, bins_j = find_bins(ps, pixel_size, kernel_size)
                        hist = torch.histogramdd(torch.from_numpy(ps).float().flip(1), bins=(bins_j, bins_i), density=False).hist
                        hist = hist.T.numpy()
                        ax[i,j].imshow(hist.reshape(kernel_size, kernel_size), origin='lower', interpolation='nearest')
                        #ax[i,j].hist2d(ps[...,1], ps[...,0], bins=25)

                plt.axis('equal')
                plt.title(f"Wavelength {wavelength:.2f}, depth {depth:.2f}")
                plt.savefig(save_dir + '/psfs_optim/' + f"heatmap_w_{wavelength:.2f}_depth_{depth:.2f}.png", format='png')

                plt.close()

        np.save(save_dir +'/psfs_optim/' + f'psf_500.npy', saved_data)
        
    if usecase == "minimize_psf":
        N = 10000
        N = 58
        max_angle = 45.
        wavelengths_list = [450., 520.0, 650.]
        #wavelengths_list = [520.0]

        base_value = lenses[1].d_sensor
        depth_list = np.arange(0.58, 0.805, 0.005)
        depth_list = np.arange(-1, 1.1, 0.1)
        depth_list = np.arange(0.12, 0.17, 0.005)

        line_pos_list = np.linspace(2.5, -2.5, 7)
        col_pos_list = np.linspace(2.5, -2.5, 7)
    
        
        saved_data = np.zeros((len(depth_list), len(line_pos_list)*len(col_pos_list)*len(wavelengths_list)))
        z_pos = 0
        for d, depth in enumerate(depth_list):
            ind = 0
            for wavelength in wavelengths_list:
                lenses[1].d_sensor = base_value + depth
                for l, line_pos in enumerate(line_pos_list):
                    for c, col_pos in enumerate(col_pos_list):
                        source_pos = np.array([col_pos, line_pos])
                        z = max(np.linalg.norm(source_pos - np.array([-12.7, 0])), np.linalg.norm(source_pos - np.array([12.7, 0])), np.linalg.norm(source_pos - np.array([0, -12.7])), np.linalg.norm(source_pos - np.array([0, 12.7])))
                        x, y = draw_circle_hexapolar(N, z, max_angle, center_x = 0, center_y = 0)
                        dist = np.stack((x, y, 70.120*np.ones(x.shape[0])), axis=-1) #- np.array([line_pos, col_pos, z_pos])


                        ps = trace_psf_from_point_source(lenses = lenses, angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = z_pos, wavelength = wavelength,
                                        normalize = False, show_rays = False, save_dir=save_dir, d = dist, ignore_invalid = False, show_res = False)
                        #saved_data[d, ind] = np.var(ps.numpy())
                        saved_data[d, ind] = np.mean(np.var(ps.numpy(), axis=0))
                        ind+=1
            print(f"Depth {depth:.2f} done")
        plt.plot(depth_list, np.mean(saved_data, axis=-1))
        plt.show()
    
    if usecase == "minimise_psf_random":
        N = 10000
        N = 58
        max_angle = 22.5
        wavelengths_list = [450., 500., 520.0, 600., 650.]
        #wavelengths_list = [520.0]

        base_value = lenses[1].d_sensor
        depth_list = np.arange(-0.5, -0.4, 0.01)

        n_run = 1

        for n in range(n_run):
            line_pos_list = np.random.uniform(-2.56, 2.56, size=49)
            col_pos_list = np.random.uniform(-2.56, 2.56, size=49)
            
            saved_data = np.zeros((len(depth_list), len(line_pos_list)*len(wavelengths_list), 2))
            z_pos = 0
            for d, depth in enumerate(depth_list):
                ind = 0
                for wavelength in wavelengths_list:
                    lenses[1].d_sensor = base_value + depth
                    for l in range(len(line_pos_list)):
                        line_pos = line_pos_list[l]
                        col_pos = col_pos_list[l]
                        source_pos = np.array([col_pos, line_pos])
                        z = max(np.linalg.norm(source_pos - np.array([-12.7, 0])), np.linalg.norm(source_pos - np.array([12.7, 0])), np.linalg.norm(source_pos - np.array([0, -12.7])), np.linalg.norm(source_pos - np.array([0, 12.7])))
                        x, y = draw_circle_hexapolar(N, z, max_angle, center_x = 0, center_y = 0)

                        dist = np.stack((x, y, 70.120*np.ones(x.shape[0])), axis=-1) #- np.array([line_pos, col_pos, z_pos])

                        ps = trace_psf_from_point_source(lenses = lenses, angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = z_pos, wavelength = wavelength,
                                        normalize = False, show_rays = False, save_dir=save_dir, d = dist, ignore_invalid = False, show_res = False)
                        #saved_data[d, ind] = np.var(ps.numpy())
                        saved_data[d, ind, :] = np.var(ps.numpy(), axis=0)
                        ind+=1
                print(f"Depth {depth:.3f} done")
            np.save(save_dir +'/psfs_optim_random/' + f'vars_{n}.npy', saved_data)
            plt.figure()
            plt.plot(depth_list, np.mean(saved_data[:,:,0], axis=-1))
            plt.title("Variance w.r.t. z for y dimension")
            #plt.savefig(save_dir + '/psfs_optim_random/' + f"vars_y_graph_{n}.png", format='png')
            plt.figure()
            
            plt.plot(depth_list, np.mean(saved_data[:,:,1], axis=-1))
            plt.title("Variance w.r.t. z for x dimension")
            #plt.savefig(save_dir + '/psfs_optim_random/' + f"vars_x_graph_{n}.png", format='png')
            #plt.close()
            plt.show()
            print(f"Run {n+1} done")
        