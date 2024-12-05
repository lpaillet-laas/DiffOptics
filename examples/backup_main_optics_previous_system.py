import os
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../")
import diffoptics as do

import time
from utils import *
import yaml

import cProfile
from main_class import *
from matplotlib import cm

# initialize a lens
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './render_pattern_demo/'

d_R = 12.7                                          # 1/2*Height of the optical objects (in mm)
d = 100.0                                           # Origin of the optical objects
angle = 53.4/2                                      # Angle of the prism

d_x1 = 1.767909                                     # Abscissa of the foot of the first curvature
d_x2 = 4.531195                                     # Abscissa of the foot of the second curvature
d_x3 = 8.651783                                     # Abscissa of the foot of the third curvature

curv_x1 = 0.0                                       # Abscissa of the first curvature midpoint
curv_x2 = 7                                         # Abscissa of the second curvature midpoint
curv_x3 = 9.5                                       # Abscissa of the third curvature midpoint

x1_e = d_x1                                         # Distance between the curvature midpoint and the foot of the first curvature
x2_e = 7-d_x2                                       # Distance between the curvature midpoint and the foot of the second curvature
x3_e = 9.5-d_x3                                     # Distance between the curvature midpoint and the foot of the third curvature
d_length = curv_x3 - curv_x1                        # Length of the lens

d_H = d_R*2                                         # Height of the prism
d_prism_length = 2*d_H*np.tan(angle*np.pi/180)      # Length of the prism

d_F = 75                                            # Focal length of the lens
d_back_F = 70.120                                   # Back focal length of the lens

lens_setup1 = [{'type': 'Aspheric',
                'params': {
                    'c': 1/((d_R**2 + x3_e**2)/(2*x3_e)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': d_back_F,
                'R': d_R
                },
                {'type': 'Aspheric',
                'params': {
                    'c': 1/((d_R**2 + x2_e**2)/(2*x2_e)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': curv_x3-curv_x2,
                'R': d_R
                },
                {'type': 'Aspheric',
                'params': {
                    'c': -1/((d_R**2 + d_x1**2)/(2*d_x1)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': curv_x2 - curv_x1,
                'R': d_R
                    },
                ]

lens_materials1 = ['air', 'sf5', 'N-BK7', 'air']

prism_test_setup = [{'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,0],
                            },
                        'd': d_back_F + d_length + d_F,
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

prism_test_materials = ['air', 'N-SK2', 'N-SF4', 'N-SK2', 'air']

angle1 = 4.49 
angle2 = 22.720 - angle1
angle3 = 2*18.1 - angle2
angle4 = 22.720 - angle3

d_prism_length = d_R*(np.tan(angle1*np.pi/180) + 2*np.tan(angle2*np.pi/180) + 2*np.tan(angle3*np.pi/180) + np.tan(angle4*np.pi/180)).item()      # Length of the prism

amici_prism_setup = [{'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,-np.tan(angle1*np.pi/180).item()],
                            },
                        'd': d_back_F + d_length + d_F,
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,np.tan(angle2*np.pi/180).item()],
                            },
                        'd': d_R*(np.tan(angle2*np.pi/180) + np.abs(np.tan(angle1*np.pi/180))).item(),
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,-np.tan(angle3*np.pi/180).item()],
                            },
                        'd': d_R*(np.tan(angle2*np.pi/180) + np.abs(np.tan(angle3*np.pi/180))).item(),
                        'R': d_R
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,np.tan(angle4*np.pi/180).item()],
                            },
                        'd': d_R*(np.tan(angle3*np.pi/180)+np.tan(angle4*np.pi/180)).item(),
                        'R': d_R
                    }]

lens_setup2 = [{'type': 'Aspheric',
                'params': {
                    'c': 1/((d_R**2 + d_x1**2)/(2*d_x1)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': d_back_F + d_length + d_F + d_prism_length + d_F,
                'R': d_R
                    },
                {'type': 'Aspheric',
                'params': {
                    'c': - 1/((d_R**2 + x2_e**2)/(2*x2_e)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': curv_x2-curv_x1,
                'R': d_R
                },
                {'type': 'Aspheric',
                'params': {
                    'c': - 1/((d_R**2 + x3_e**2)/(2*x3_e)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': curv_x3 - curv_x2,
                'R': d_R
                }
                ]


lens_materials2 = ['air', 'N-BK7', 'sf5', 'air']

""" lens_setup1 = [{'type': 'Aspheric',
                'params': {
                    'c': 1/((d_R**2 + x3_e**2)/(2*x3_e)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': d_back_F,
                'R': d_R
                },
                {'type': 'Aspheric',
                'params': {
                    'c': 1/((d_R**2 + x2_e**2)/(2*x2_e)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': curv_x3-curv_x2,
                'R': d_R
                },
                {'type': 'Aspheric',
                'params': {
                    'c': -1/((d_R**2 + d_x1**2)/(2*d_x1)),
                    'k': 0.0,
                    'ai': None,
                },
                'd': curv_x2 - curv_x1,
                'R': d_R
                }
                ]
lens_materials1 = ['air', 'sf5', 'N-BK7', 'air'] """

""" perfect_lens = [{'type': 'ThinLens',
                 'params': {
                     'f': 30.0,
                 },
                 'd': d_back_F + d_length + d_F + d_F/2,
                 'R': d_R,

                }]

perfect_lens_materials2 = ['air', 'air'] """

d_tilt_angle_test2 = - 9.088 # if middle w is 520nm
d_shift_value_test2 = 1.1

d_tilt_angle2 = 4.902 - angle4
d_tilt_angle2 *= -1
d_tilt_angle2 -= 0.021

d_shift_value2 = 0.0

if __name__ == '__main__':
    adjustment = 2.56

    usecase = 'compare_positions_trace'
    usecase = 'save_pos_render'
    usecase = 'optimize_psf_zemax'
    usecase = 'get_dispersion'
    usecase = 'render'
    usecase = 'psf_line'
    usecase = 'kfk'
    #usecase = 'render_lots'
    #usecase = 'compare_psf_zemax'

    #usecase = 'compare_positions_trace_back'


    oversample = 100
    
    d_last_surface_distance = d_back_F + d_length + d_F + d_prism_length
    d_center_prism = d_back_F + d_length + d_F + d_prism_length/2


    #optimized_lens_shift = - 0.42
    optimized_lens_shift = -0.03
    list_d_sensor = [d_back_F + d_length + d_F,
                     d_back_F + d_length + d_F + d_prism_length + d_F,
                     d_back_F + d_length + d_F + d_prism_length + d_F + d_length + d_back_F + optimized_lens_shift]
    list_r_last = [d_R, d_R, d_R]

    list_film_size = [[adjustment*100,adjustment*100] for i in range(3)]
    list_pixel_size = [1/adjustment*80.0e-3]*3
    list_pixel_size = [1/3*1000/28*1e-3]*3
    list_theta_y = [0., 0., d_tilt_angle2]
    list_theta_x = [0., 0., 0.]
    list_theta_z = [0., 0., 0.]
    list_origin = [None, [0., 0., d_center_prism], [0.,0.,d_last_surface_distance]]
    list_shift = [None, [0., 0., -d_center_prism], [d_shift_value2,0.,-d_last_surface_distance]]
    system_wavelengths = torch.linspace(450, 650, 28*oversample)

    lens_group = HSSystem(list_systems_surfaces=[lens_setup1, amici_prism_setup, lens_setup2], list_systems_materials=[lens_materials1, prism_test_materials, lens_materials2],
                        list_d_sensor=list_d_sensor, list_r_last=list_r_last, list_film_size=list_film_size, list_pixel_size=list_pixel_size,
                        list_theta_y=list_theta_y, list_theta_x=list_theta_x, list_theta_z=list_theta_z,
                        list_origin=list_origin, list_shift=list_shift,
                        wavelengths = system_wavelengths,
                        device=device, save_dir="./render_pattern_demo/")

    lens_group.export_system("system.yml")
    lens_group.d_subsystems = torch.tensor([0, 0, 0])
    lens_group.update_system()
    #lens_group.system = [lens_group.system[0], lens_group.system[-1]]
    #lens_group.size_system = 1

    lens_group.combined_plot_setup()

    if usecase in ['psf', 'spot']:   
        lens_group.system[0].d_sensor = d_back_F + d_length + 0.5  # For plotting purposes
        lens_group.system[1].d_sensor = d_back_F + d_length + d_F + d_prism_length + 0.5  # For plotting purposes

    
    if usecase == 'spot':
        wavelengths = torch.tensor([450, 550, 650])
        lens_group.plot_spot_less_points(400, 20*1e-3, wavelengths = wavelengths)
    elif usecase == 'test':
        N = 10
        wavelength = torch.tensor([520.0])
        x_pos, y_pos, z_pos = 0., 0., 0.
        angle = torch.tensor([22.5], requires_grad=True)
        
        #lens.d_sensor = torch.tensor([lens.d_sensor], requires_grad=True)
        lens_group.system[-1].d_sensor = torch.tensor([lens_group.system[-1].d_sensor], requires_grad=True)
        #lens.plot_setup2D()
        
        optimizer = torch.optim.Adam([lens_group.system[-1].d_sensor, angle], lr=1e-1)
        for i in range(100):
            optimizer.zero_grad()

            d = lens_group.extract_hexapolar_dir(N, torch.tensor([0., 0.]), angle)
            
            ps = lens_group.trace_psf_from_point_source(angles=[[0, 0]], x_pos=0, y_pos=0, z_pos=0, wavelength = wavelength,
                 show_rays = False, d = d, ignore_invalid = False, show_res = False)
            loss = -(torch.max(ps[...,0]) - torch.min(ps[...,0]))
            print("Loss: ", loss)
            loss.backward()
            optimizer.step()
            print("Angle: ", angle.clone().detach())
            print("Dist capteur: ", lens_group.system[-1].d_sensor.clone().detach())
            print("")

    elif usecase == 'spot_compare':
        lens_group.compare_spot(400, 20*1e-3, opposite = [True, False], shift = [-0.014, 0.],
                                path_compare='/home/lpaillet/Documents/Codes/simca/', model = 'simca',
                                config_path = '/home/lpaillet/Documents/Codes/simca/simca/configs/cassi_system_optim_optics_full_triplet_sd_cassi_prism_propag.yml')
    elif usecase == 'compare_positions_trace':
        N = 2
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5
        wavelength = 520.0
        lens_group.system[-1].d_sensor -= 0.040

        line_pos1, col_pos1 = 0., 0.
        line_pos2, col_pos2 = 2.5, 0.
        line_pos3, col_pos3 = -2.5, 0.
        line_pos4, col_pos4 = 0., 2.5
        line_pos5, col_pos5 = 0., -2.5
        list_source_pos = [torch.tensor([col_pos1, line_pos1]), torch.tensor([col_pos2, line_pos2]), torch.tensor([col_pos3, line_pos3]), torch.tensor([col_pos4, line_pos4]),
                           torch.tensor([col_pos5, line_pos5])]
        colors = ['b-', 'g-', 'r-', 'k-', 'm-']

        lens_group.compare_positions_trace(N, list_source_pos, max_angle, wavelength, colors=colors)

    elif usecase == 'compare_wavelength_trace':
        wavelengths = [450., 520., 650.]
        colors = ['b-', 'g-', 'r-']
        N = 2
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5
        lens_group.system[-1].d_sensor -= 0.040

        line_pos, col_pos = 0., 0.
        source_pos = torch.tensor([col_pos, line_pos])

        lens_group.compare_wavelength_trace(N, source_pos, max_angle, wavelengths, colors=colors)
    elif usecase == 'compare_positions_trace_back':
        for i, lens in enumerate(lens_group.system[::-1]):
            lens_mts_R, lens_mts_t = lens._compute_transformation().R, lens._compute_transformation().t
            if i > 0:
                lens_group.prepare_mts(-i-1, lens.pixel_size, lens.film_size, start_distance = max_z, R=lens_mts_R, t=lens_mts_t)
            else:
                max_z = lens.d_sensor*np.cos(lens.theta_y*np.pi/180).item() + lens.origin[-1] + lens.shift[-1] # Last z coordinate in absolute coordinates
                lens_group.prepare_mts(-1, lens.pixel_size, lens.film_size, start_distance = max_z, R=lens_mts_R, t=lens_mts_t)    
        
        lens_group.combined_plot_setup(with_sensor=False)
        
        lens_group.system = lens_group.system[::-1]
        lens_group.system[0].film_size[0] = 2
        lens_group.system[0].film_size[1] = 2
        lens_group.system[0].pixel_size = 2

        def sample_ray_sensor(lens, wavelength, offset=np.zeros(2), aperture_reduction = 1):
            """
            Sample rays on the sensor plane.
            """
            if not lens.mts_prepared:
                raise Exception('MTS unprepared; please call `prepare_mts()` first!')
            
            N = np.prod(lens.film_size)
            
            # sensor and aperture plane samplings
            sample2 = lens._generate_sensor_samples()
            sample3 = lens._generate_aperture_samples()
            sample3 = sample3 / aperture_reduction

            # wavelength [nm]
            wav = wavelength * np.ones(N)
            
            # sample ray
            valid, ray, oss = _sample_ray_render(lens, N, wav, sample2, sample3, offset)
            # print("Rendered ray: ", ray)
            ray_new = lens.to_world.transform_ray(ray)
            #ray_new = self.mts_Rt.transform_ray(ray)

            return valid, ray_new, oss
        def normalize(d):
            return d / torch.sqrt(torch.sum(d**2, axis=-1))[..., None]
        
        def _sample_ray_render(lens, N, wav, sample2, sample3, offset):
            """
            `offset`: sensor position offsets [mm].
            """

            # sample2 \in [ 0, 1]^2
            # sample3 \in [-1, 1]^2
            if not lens.mts_prepared:
                raise Exception('MTS unprepared; please call `prepare_mts()` first!')
            
            # normalized to [-0,5, 0.5]
            sample2 = sample2 - 0.5
            
            # sample sensor and aperture planes
            p_sensor = sample2 * np.array([
                lens.pixel_size * lens.film_size[0], lens.pixel_size * lens.film_size[1]
            ])[None,:]

            # perturb sensor position by half pixel size
            p_sensor = p_sensor + (np.random.rand(*p_sensor.shape) - 0.5) * lens.pixel_size

            # offset sensor positions
            p_sensor = p_sensor + offset

            # aperture samples (last surface plane)
            p_aperture = sample3 * lens.aperture_radius
            d_xy = p_aperture - p_sensor

            # construct ray
            #o = torch.Tensor(np.hstack((p_sensor, np.zeros((N,1)))).reshape((N,3)))
            d_sensor = lens.d_sensor.item() if isinstance(lens.d_sensor, np.ndarray) or isinstance(lens.d_sensor, torch.Tensor) else lens.d_sensor
            o = torch.Tensor(np.hstack((p_sensor, np.zeros((N,1)) + d_sensor)).reshape((N,3)))
            d = torch.Tensor(np.hstack((d_xy, lens.aperture_distance.item() * np.ones((N,1)))).reshape((N,3)))
            d = normalize(d)
            wavelength = torch.Tensor(wav).float()
            created_ray = do.Ray(o, d, wavelength, device=lens.device)
            # trace        
            valid, ray, oss = lens._trace(created_ray, record = True)
            return valid, ray, oss
        
        def render_single_back(lens_group, wavelength, screen, aperture_reduction = 1):
            """
            Renders back propagation of single ray through a series of lenses onto a screen.

            Args:
                wavelength (float): The wavelength of the light.
                screen (object): The screen object onto which the light is projected.

            Returns:
                tuple: A tuple containing the intensity values (I) and the mask indicating valid pixels on the screen.
            """
            # Sample rays from the sensor
            ####valid, ray_mid = self.system[-1].sample_ray_sensor(wavelength.item(), aperture_reduction = aperture_reduction)
            oss_list = [0 for i in range(len(lens_group.system))]
            valid, ray_mid, oss = sample_ray_sensor(lens_group.system[0], wavelength.item(), aperture_reduction = aperture_reduction)
            oss_list[0] = oss
            # ray_mid.o = ray_mid.o[valid, :]
            # ray_mid.d = ray_mid.d[valid, :]
            # print("Ray 1: ", ray_mid)
            print("Nb valid 1: ", torch.sum(valid))
            # Trace rays through each lens in the system
            
            ####for lens in self.system[::-1][1:-1]:
            for i, lens in enumerate(lens_group.system[1:-1]):
                ray_mid = lens.to_object.transform_ray(ray_mid)
                valid_1, ray_mid, oss = lens._trace(ray_mid, record=True)
                oss_list[i+1] = oss
                #ray_mid.o = ray_mid.o[valid_1, :]
                #ray_mid.d = ray_mid.d[valid_1, :]
                #ray_mid = lens.mts_Rt.transform_ray(ray_mid)
                ray_mid = lens.to_world.transform_ray(ray_mid)
                #print(f"Ray mid: ", ray_mid)
                print("Nb valid mid: ", torch.sum(valid_1))
                valid = valid & valid_1
                
            # Trace rays to the first lens
            ####ray_mid = self.system[0].to_object.transform_ray(ray_mid)
            ray_mid = lens_group.system[-1].to_object.transform_ray(ray_mid)
            ####valid_last, ray_last = self.system[0]._trace(ray_mid)
            valid_last, ray_last, oss = lens_group.system[-1]._trace(ray_mid, record = True)
            oss_list[-1] = oss
            print("Nb valid last: ", torch.sum(valid_last))
            valid_last = valid & valid_last
            #print("Ray last before transform: ", ray_last)

            #ray_last = self.system[0].mts_Rt.transform_ray(ray_last)
            ####ray_last = self.system[0].to_world.transform_ray(ray_last)
            ray_last = lens_group.system[-1].to_world.transform_ray(ray_last)
            #print("Ray last: ", ray_last)
            
            I = mask = None
            """ # Intersect rays with the screen
            uv, valid_screen = screen.intersect(ray_last)[1:]
            # Apply mask to filter out invalid rays
            mask = valid_last & valid_screen
            
            # Calculate intensity values on the screen
            I = screen.shading(uv, mask) """
            
            return I, mask, oss_list

        I, mask, oss_list = render_single_back(lens_group, torch.tensor([520.]), None)

        for i, oss in enumerate(oss_list):
            for os in oss:
                for o in os:
                    os = lens_group.system[i].to_world.transform_point(torch.Tensor(np.asarray(os))).cpu().detach().numpy()

        ax, fig = lens_group.plot_setup_with_rays(oss_list, ax=None, fig=None, show=False)
        plt.show()
    elif usecase == 'render':
        wavelengths = torch.tensor(system_wavelengths).float().to(device)

        nb_rays = 10

        texture = np.kron([[1, 0] * 10, [0, 1] * 10] * 10, np.ones((25, 25)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 25x25 on a 500x500 size
        texture = np.ones((256, 256, len(wavelengths)), dtype=np.float32)

        #texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556].astype('float32')
        
        texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
        texture = 1000*torch.tensor(texture).repeat_interleave(oversample).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[:, 128] = 1
        texture = np.multiply(texture, mask[:,:,np.newaxis])      
        
        texture = torch.from_numpy(texture).float().to(device)

        z0 = list_d_sensor[-1]
        #offsets = -d_R + d_x1
        print("z0 ", z0)
        image = lens_group.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                        texture=texture, offsets=[0, 0, 0], aperture_reduction=1, plot=True)
        
        torch.save(image, f'test_n{nb_rays}_ov{int(len(wavelengths)//28)}.pt')

    elif usecase == 'render_lots':
        wavelengths = torch.tensor(system_wavelengths).float().to(device)

        for n in range(1,21):
            nb_rays = 20*n

            texture = np.kron([[1, 0] * 10, [0, 1] * 10] * 10, np.ones((25, 25)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 25x25 on a 500x500 size
            texture = np.ones((256, 256, len(wavelengths)), dtype=np.float32)

            #texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556].astype('float32')
            
            texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
            texture = 1000*torch.tensor(texture).repeat_interleave(oversample).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
            mask = np.zeros((256, 256), dtype=np.float32)
            mask[:, 128] = 1
            texture = np.multiply(texture, mask[:,:,np.newaxis])      
            
            texture = torch.from_numpy(texture).float().to(device)

            z0 = list_d_sensor[-1]
            #offsets = -d_R + d_x1
            image = lens_group.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                            texture=texture, offsets=[0, 0, 0], aperture_reduction=1, plot=False)
            
            torch.save(image, f'test_n{nb_rays}_ov{int(len(wavelengths)//28)}.pt')
    
    elif usecase == 'save_pos_render':
        wavelengths = torch.tensor(system_wavelengths).float().to(device)
        nb_rays = 40

        texture = np.kron([[1, 0] * 8, [0, 1] * 8] * 8, np.ones((16, 16)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 16x16 on a 256x256 size
        texture = np.ones((int(list_film_size[0][0]), int(list_film_size[0][1]), len(wavelengths)), dtype=np.float32)
        texture = texture[:256,:256,0]
        texture = torch.from_numpy(texture).float().to(device)

        z0 = list_d_sensor[-1]
        nb_squares = int(list_film_size[0][0])//32
        nb_squares = 8 
        
        big_uv, big_mask = lens_group.save_pos_render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                          texture=texture, offsets=[0, 0, 0], aperture_reduction=4)
        
        texture = np.kron([[1, 0] * nb_squares, [0, 1] * nb_squares] * nb_squares, np.ones((16, 16)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 16x16 on a 256x256 size
        texture = texture[:256,:256,:]

        texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556].astype('float32')
        
        texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
        texture = 1000*torch.tensor(texture).repeat_interleave(4).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[:, 128] = 1
        texture = np.multiply(texture, mask[:,:,np.newaxis])      
        
        texture = torch.from_numpy(texture).float().to(device)
        #texture = np.zeros((int(list_film_size[0][0]), int(list_film_size[0][1]), len(wavelengths)), dtype=np.float32)
        #texture[:, texture.shape[0]//2, :] = 1

        image = lens_group.render_based_on_saved_pos(big_uv = big_uv, big_mask = big_mask, texture = texture, nb_rays=nb_rays, wavelengths = wavelengths,
                                    z0=z0, plot = True)
        torch.save(image, 'test.pt')

    elif usecase == 'get_dispersion':
        wavelengths = torch.linspace(450, 650, 3)
        pos_dispersed, pixel_dispersed = lens_group.central_positions_wavelengths(wavelengths)
        rounded_pixel = pixel_dispersed.round().int()
        print(rounded_pixel)
        empty_space = torch.arange((512))
        reduced_empty = empty_space[512//4 + rounded_pixel[0,0]: 512//4 + 512//2 + rounded_pixel[-1,0]]
        print(reduced_empty.shape)
        rounded_pixel = rounded_pixel - rounded_pixel.min(dim=0).values
        print(pos_dispersed, rounded_pixel)
        fake_cube = torch.zeros(256, 3)
        for i in range(3):
            fake_cube[:, i]=reduced_empty[rounded_pixel[i,0]:rounded_pixel[i,0]+256]
        print(fake_cube[:,-1])

    elif usecase == 'psf':
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5
        wavelength = 520.0

        line_pos, col_pos = 0., 0.
        source_pos = torch.tensor([col_pos, line_pos])

        ps = lens_group.plot_psf(N, source_pos, max_angle, wavelength, show_rays = False, show_res = False)
    
    elif usecase == 'psf_line':
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5
        kernel_size = 11
        wavelengths = system_wavelengths

        line_pos, col_pos = 0., 0.
        source_pos = torch.tensor([col_pos, line_pos])

        d = lens_group.extract_hexapolar_dir(N, source_pos, max_angle)

        psf_line = torch.empty((len(wavelengths), kernel_size, kernel_size))

        for w_id, wavelength in enumerate(tqdm(wavelengths)):
            ps = lens_group.trace_psf_from_point_source(angles = None, x_pos = source_pos[0], y_pos = source_pos[1], z_pos = 0., wavelength = wavelength,
                        show_rays = False, d = d, ignore_invalid = False, show_res = False)
            bins_i, bins_j, centroid = find_bins(ps, list_pixel_size[0], kernel_size, same_grid=False)
            hist_ps = torch.histogramdd(ps, bins=(bins_i, bins_j), density=False).hist
            hist_ps /= hist_ps.sum()
            psf_line[w_id, :, :] = hist_ps.reshape(kernel_size, kernel_size)
            #plt.imshow(hist_ps, origin='lower', interpolation='nearest')
            #plt.show()
        torch.save(psf_line, 'psf_line.pt')        
        
    elif usecase == 'optimize_adam_psf_zemax':
        n_iter = 100
        start_dist = lens_group.system[-1].d_sensor
        N = 20
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        pixel_size = 5e-3

        source_pos1 = torch.tensor([0., 0.])
        source_pos2 = torch.tensor([2.5, 0.])
        source_pos3 = torch.tensor([0., 2.5])
        source_pos4 = torch.tensor([2.5, 2.5])
        source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
        w_list = [520.0, 450., 650.]

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/AMICI/ray_positions_wavelength_W1_field_F1.h5"

        params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                  for i in range(len(source_pos_list)) for j in range(len(w_list))]
        
        lens_group.system[-1].d_sensor = torch.tensor([lens_group.system[-1].d_sensor], requires_grad=True)
        angle = torch.tensor([20.], requires_grad=True)
        
        optimizer = torch.optim.Adam([lens_group.system[-1].d_sensor, angle], lr=1e-2)
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = 0.0
            for k in range(len(params)):
                d = lens_group.extract_hexapolar_dir(N, params[k][0], angle) 
                ps = lens_group.trace_psf_from_point_source(angles = None, x_pos = params[k][0][0], y_pos = params[k][0][1], z_pos = 0., wavelength = params[k][1],
                        show_rays = False, d = d, ignore_invalid = False, show_res = False).float()
                bins_i, bins_j, centroid = find_bins(ps, pixel_size, 11)
                hist_ps = torch.histogramdd(ps, bins=(bins_i, bins_j), density=False).hist
                hist_ps /= hist_ps.sum()

                ps_zemax = params[k][2]
                ps_zemax = torch.stack(ps_zemax, dim=-1).float()
                bins_i, bins_j, centroid_zemax = find_bins(ps_zemax, pixel_size, 11)
                hist_zemax = torch.histogramdd(ps_zemax, bins=(bins_i, bins_j), density=False).hist
                hist_zemax /= hist_zemax.sum()
                
                #loss += torch.sqrt(torch.mean(((hist_ps - hist_zemax))**2))
                loss += torch.abs(centroid_zemax - centroid).sum()
                #loss += torch.abs(torch.max(ps, dim=0).values - torch.max(ps_zemax, dim=0).values).sum()
            loss /= len(params)
            loss.backward()
            optimizer.step()
            print("Loss: ", loss)
            print("Angle: ", angle.clone().detach())
            print("Dist capteur: ", lens_group.system[-1].d_sensor.clone().detach())
            print("")
            
    elif usecase == 'optimize_psf_zemax':
        start_dist = lens_group.system[-1].d_sensor
        N = 30
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        pixel_size = 5e-3

        source_pos1 = torch.tensor([0., 0.])
        source_pos2 = torch.tensor([2.5, 0.])
        source_pos3 = torch.tensor([0., 2.5])
        source_pos4 = torch.tensor([2.5, 2.5])
        source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
        w_list = [520.0, 450., 650.]

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/AMICI/ray_positions_wavelength_W1_field_F1.h5"

        depth_list = torch.from_numpy(np.arange(-2., 2., 0.01))
        angle_list = torch.from_numpy(np.arange(5, 50, 2))
        angle_list = torch.tensor([22.5])
        depth_list = torch.tensor([0.0])

        params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                  for i in range(len(source_pos_list)) for j in range(len(w_list))]

        map = lens_group.fit_psf(N, params, depth_list, angle_list, start_dist, pixel_size, kernel_size = 11, show_psf=True)
        torch.save(map, 'map_optim.pt')
        if map.shape[0] > 1 and map.shape[1] > 1:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            X, Y = np.meshgrid(angle_list, depth_list)

            surf = ax.plot_surface(X, Y, map, cmap=cm.coolwarm, linewidth=0, antialiased=False)

            fig.colorbar(surf, shrink=0.5, aspect=5)
        else:
            if map.shape[0] > 1:
                plt.plot(depth_list, map)
            else:
                plt.plot(angle_list, map)
        plt.show()


    elif usecase == 'compare_psf_zemax':
        N = 30
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5
        wavelength = 520.0

        pixel_size = 5e-3

        source_pos1 = torch.tensor([0., 0.])
        source_pos2 = torch.tensor([2.5, 0.])
        source_pos3 = torch.tensor([0., 2.5])
        source_pos4 = torch.tensor([2.5, 2.5])
        source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
        w_list = [520.0, 450., 650.]

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/AMICI/ray_positions_wavelength_W1_field_F1.h5"

        params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                  for i in range(len(source_pos_list)) for j in range(len(w_list))]


        lens_group.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)
        

    elif usecase == "psf_train_database":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5

        lim = 2.7
        nb_pts = 50
        min_w = 450
        max_w = 650
        n_w = 55
        line_pos_list = np.linspace(-lim, lim, nb_pts)
        col_pos_list = np.linspace(-lim, lim, nb_pts)
        wavelength_list = np.linspace(min_w, max_w, n_w)

        lens_group.create_train_database(N, line_pos_list, col_pos_list, max_angle, wavelength_list, lim)

    elif usecase == "psf_test_database":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5

        database_size = 4000
        lim = 2.56
        line_pos_list = np.random.uniform(-lim, lim, size=database_size)
        col_pos_list = np.random.uniform(-lim, lim, size=database_size)
        wavelength_list = np.random.randint(0, 54, size=database_size)

        lens_group.create_test_database(N, line_pos_list, col_pos_list, max_angle, wavelength_list, lim, index_save = True)

    elif usecase == "psf_test_regular_database":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5

        lim = 2.7
        nb_pts = 50
        min_w = 450
        max_w = 650
        n_w = 55
        line_pos_list = np.linspace(-lim, lim, nb_pts)[:-1] + 2*lim/(nb_pts-1)/2
        col_pos_list = np.linspace(-lim, lim, nb_pts)[:-1] + 2*lim/(nb_pts-1)/2
        wavelength_list = np.linspace(min_w, max_w, n_w)

        lens_group.create_test_regular_database(N, line_pos_list, col_pos_list, max_angle, wavelength_list, 2.56, reduced = True, index_save = True)

    elif usecase == "view_psf_field":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        max_angle = 22.5
        wavelength_list = [450., 520., 650.]

        depth_list = np.arange(-0.5, 0.51, 0.05)
        depth_list = [0]

        nb_pts = 3
        line_pos_list = np.linspace(2.5, -2.5, nb_pts)
        col_pos_list = np.linspace(2.5, -2.5, nb_pts)

        lens_group.psf_field_view(N, depth_list, line_pos_list, col_pos_list, max_angle, wavelength_list)

    elif usecase == "minimize_psf":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5

        wavelength_list = [450., 520.0, 650.]

        depth_list = np.arange(-0.025, 0.025, 0.005)

        nb_pts = 7

        line_pos_list = np.linspace(2.5, -2.5, nb_pts)
        col_pos_list = np.linspace(2.5, -2.5, nb_pts)

        lens_group.minimize_psf(N, depth_list, line_pos_list, col_pos_list, max_angle, wavelength_list)
    
    elif usecase == "minimize_psf_random":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 22.5

        wavelength_list = [450., 500., 520.0, 600., 650.]

        lim = 2.56
        nb_pts = 49
        depth_list = np.arange(-0.08, 0.02, 0.01)

        n_run = 1

        lens_group.minimize_psf_random(N, depth_list, max_angle, wavelength_list, lim, nb_pts, n_run)

def extract_acq_from_cube(cube_3d, dispersion_pixels, middle_pos, texture_size):
        acq_2d = cube_3d.sum(-1)
        rounded_disp = dispersion_pixels.round().int() # Pixels to indices

        return acq_2d[middle_pos[0] - texture_size[0]//2 + rounded_disp[0,0]: middle_pos[0] + texture_size[0]//2 + rounded_disp[-1,0],
                        middle_pos[1] - texture_size[1]//2 + rounded_disp[0,1]: middle_pos[1] + texture_size[1]//2 + rounded_disp[-1,1]]

def shift_back(inputs, dispersion_pixels, n_lambda = 28): 
    """
        Input [bs, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H, W]
    """
    bs, H, W = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    output = torch.zeros(bs, n_lambda, H - max_min[0], W - max_min[1])
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(n_lambda):
        output[:, i, :, :] = inputs[:, abs_shift[i, 0]: H - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: W - max_min[1] + abs_shift[i, 1]]
    return output

def shift(inputs, dispersion_pixels, n_lambda = 28):
    """
        Input [bs, n_wav, H, W]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
    """
    bs, n_lambda, H, W = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    output = torch.zeros(bs, n_lambda, H + max_min[0], W + max_min[1])
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(n_lambda):
        output[:, i, abs_shift[i, 0]: H + abs_shift[i, 0], abs_shift[i, 1]: W + abs_shift[i, 1]] = inputs[:, i, :, :]
    return output

oversample = 100

""" image_20_4 = torch.load("test_n20_ov4.pt").flip(0,1)
image_180 = torch.load("test_n180_ov4.pt").flip(0,1)
image_20_5 = torch.load("test_n20_ov5.pt").flip(0,1) """

image_20_10 = torch.load("test_n20_ov10.pt").flip(0,1)

image_160_10 = torch.load("test_n160_ov10.pt").flip(0,1)



texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556]
texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
texture = 1000*torch.tensor(texture).repeat_interleave(oversample).float()
# plt.figure()
# plt.imshow(np.sum(texture, axis=-1))
image = torch.load("test_single_n20_ov1.pt").flip(0,1)
plt.figure()
plt.title("Panchro acquisition")
plt.imshow(torch.sum(image, dim=-1))

"""plt.figure()
plt.plot(torch.mean(torch.sum(image, dim=-1)[110:146, 81:172], dim=0))
plt.xlabel("Pixel")
plt.title("Spread spectrum") """
plt.figure()
plt.plot(lens_group.wavelengths, texture)
plt.xlabel("Wavelength")
plt.title("True spectrum")
psf_line = torch.load('psf_line.pt')
convoluted_spectrum = torch.zeros(28*oversample+5+5)
print(psf_line.shape)
for i in range(28*oversample):
    convoluted_spectrum[i:i+11] += texture.flip(0)[i]*psf_line[i, 5, :]   # Flip because rendered image has to be flipped
    #convoluted_spectrum[i:i+11] += torch.sum(texture[i]*psf_line[i, :, :], dim=0)
convoluted_spectrum = convoluted_spectrum.flip(0)


x_axis = 1/lens_group.pos_dispersed[:,0].diff()
convoluted_with_dispersion = x_axis*convoluted_spectrum[6:-5]

dispersed_texture = texture[1:]*1/lens_group.pos_dispersed[:,0].diff()

plt.figure()
plt.plot(lens_group.pos_dispersed[1:,0], dispersed_texture)
plt.xlabel("Position relative to center [mm]")
plt.title("Dispersion applied to spectrum")

dispersed_texture_pixel = torch.zeros(lens_group.pixel_dispersed.floor().unique().shape[0]+5+5)
#dispersed_texture_pixel = torch.zeros(lens_group.pixel_dispersed.floor().unique().shape[0])
dispersed_pixel_count = torch.zeros(lens_group.pixel_dispersed.floor().unique().shape[0])
ind = 5
step = lens_group.pixel_dispersed.floor().int()[0,1]
accu = 0
psf_solo = torch.sum(psf_line[psf_line.shape[0]//2, :, :], axis=0)
for i, elem in enumerate(texture):
    if lens_group.pixel_dispersed.floor().int()[i,1] == step:
        accu += elem*torch.sum(psf_line[i, :, :], axis=0)
        #accu += elem*psf_solo
        dispersed_pixel_count[ind-5] += 1
        #accu += elem
    else:
        #dispersed_texture_pixel[ind] = accu
        dispersed_texture_pixel[ind-5:ind+6] += accu
        accu = elem*torch.sum(psf_line[i, :, :], axis=0)
        #accu = elem*psf_solo
        #accu = elem
        step = lens_group.pixel_dispersed.floor().int()[i,1]
        ind += 1
        dispersed_pixel_count[ind-5] = 1
#dispersed_texture_pixel[ind] = accu
dispersed_texture_pixel[ind-5:ind+6] += accu
print(dispersed_pixel_count)

plt.figure()
#plt.plot(lens_group.pixel_dispersed.floor().int()[:,1].unique(), dispersed_texture_pixel)
plt.plot(lens_group.pixel_dispersed.floor().int()[:,1].unique(), dispersed_texture_pixel[5:-5])
plt.xlabel("Pixels relative to center [px]")
plt.title("Dispersion + PSF applied to pixelated spectrum")


"""plt.figure()
plt.plot(torch.sum(image_20_4, dim=-1)[128, 81:172])
plt.xlabel("Pixel")
plt.title("20 rays, oversampling 4")
plt.figure()
plt.plot(torch.sum(image_20_5, dim=-1)[128, 81:172])
plt.xlabel("Pixel")
plt.title("20 rays, oversampling 5")
plt.figure()
plt.plot(torch.sum(image_180, dim=-1)[128, 81:172])
plt.xlabel("Pixel")
plt.title("180 rays, oversampling 4")
plt.figure()
plt.plot(torch.sum(image_180, dim=-1)[128, 81:172])
plt.xlabel("Pixel")
plt.title("180 rays, oversampling 10") """

plt.figure()
#plt.plot(torch.mean(torch.sum(image_160_10, dim=-1)[128-5:128+5, 81:172], dim=0))
plt.plot(torch.sum(image_160_10, dim=-1)[129, 81:172])
plt.xlabel("Pixel")
plt.title("160 rays, oversampling 10")
plt.show()





""" texture = torch.empty(256, 256, 3)
res = extract_acq_from_cube(image, lens_group.pixel_dispersed, (torch.tensor(list_film_size[-1])//2).int().flip(0), (torch.tensor(texture.shape[:2])).int().flip(0))
plt.imshow(res)
plt.show()
res = res.unsqueeze(0)
back_res = shift_back(res, lens_group.pixel_dispersed, n_lambda = 28)
print(back_res.shape) """
