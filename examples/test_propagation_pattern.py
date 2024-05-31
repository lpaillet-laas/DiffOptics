import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg

import sys
sys.path.append("../")
import diffoptics as do

from tqdm import tqdm

import time

# initialize a lens
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lens = do.Lensgroup(device=device)

save_dir = './render_pattern_demo/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

### Optical Design #############################################################################################################################

#R = 12.7                  # 1/2*Height of the optical objects (in mm)
R = 12.5                  # 1/2*Height of the optical objects (in mm)
d = 120.0
angle = 53.4/2            # Angle of the prism

x1 = R                    # x1 is the distance from the origin (d) to the first prism
H = R*2                   # H is the height of the prism
x2 = x1 + 2*H*np.tan(angle*np.pi/180)  # x2 is the distance from the origin (d) to the end of the prism structure

tc = 4.1                 # Thickness of the lens + curvature
te = 2                   # Thickness of the lens
e = tc-te                # Thickness of the curvature
x0 = x1                  # x0 is the distance from the origin (d) to the first lens BETTER SET AT x1

prism_length = 2*H*np.tan(angle*np.pi/180)   # Length of the prism structure

F = 75                   # Focal length of the lens (in mm)

offset_collim = 2.4

# Coefs for the surfaces
coefs = torch.tensor([[0, 0, 0, 0, 0, 0],                                              # Start of lens 1
                      [x0+e, 0, 0, -(R+e-x0)/(x0*x0), 0, -(R+e-x0)/(x0*x0)],           # End of lens 1
                      [0, 0, 0, 0, 0, 0],                                              # Prism 1
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],                        # Prism 2
                      [0, 0, -np.tan(angle*np.pi/180), 0, 0, 0],                       # Prism 3
                      [0, 0, 0, 0, 0, 0],                                              # End of prism
                      [0, 0, np.tan(10*np.pi/180), 0, 0, 0],                           # Ghost surface
                      [x0-e, 0, 0, (R+e-x0)/(x0*x0), 0, (R+e-x0)/(x0*x0)],             # Start of lens 2
                      [0, 0, 0, 0, 0, 0]]).float()                                     # End of lens 2

# Surfaces definition
surfaces = [
    do.XYPolynomial(R, F - te, J=1, ai=coefs[0][:3], device = device),                    # Start of lens 1
    do.XYPolynomial(R, F - x0, J=2, ai=coefs[1], device = device),                    # End of lens 1
    do.XYPolynomial(R, 2*F + offset_collim - prism_length/2, J=1, ai=coefs[2][:3], device = device),                                                       # Prism 1
    do.XYPolynomial(R, 2*F + offset_collim - prism_length/2 + R*np.tan(angle*np.pi/180), J=1, ai=coefs[3][:3], device = device),                           # Prism 2
    do.XYPolynomial(R, 2*F + offset_collim - prism_length/2 + (H+R)*np.tan(angle*np.pi/180), J=1, ai=coefs[4][:3], device = device),                       # Prism 3
    do.XYPolynomial(R, 2*F + offset_collim + prism_length/2, J=1, ai=coefs[5][:3], device = device),                         # End of prism
#    do.XYPolynomial(R, d + 10 + 2*H*np.tan(angle*np.pi/180), J=1, ai=coefs[6][:3], device = device),                    # Ghost surface
    do.XYPolynomial(R, 3*F + 2*offset_collim - x0, J=2, ai=coefs[7], device = device),           # Start of lens 2
    do.XYPolynomial(R, 3*F + 2*offset_collim + te, J=1, ai=coefs[8][:3], device = device),       # End of lens 2
]

# Materials definition
materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air'),
    do.Material('N-SK2'),
    do.Material('N-SF4'),
    do.Material('N-SK2'),
    do.Material('air'),
#    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]

# Load the surfaces and materials into the lens + last parameters to set
lens.load(surfaces, materials)
lens.d_sensor = 4*F + 2*offset_collim #+tc # distance to sensor from 0 (in mm?)
lens.r_last = R             # radius of sensor (in mm?) in plot, changed in prepare_mts

lens.film_size = [131, 131] # [pixels]
lens.pixel_size = 80.0e-3*3   # [mm]

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

d = 2*F - prism_length/2

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


d_surfaces = [
    do.XYPolynomial(d_R, d_F - d_length/2 + d_x1-d_x0, J=2, ai=d_coefs[0], device = device),              # Start of doublet 1                                                     # Start of doublet 1
    do.XYPolynomial(d_R, d_F - d_length/2 + d_x2-d_x0, J=2, ai=d_coefs[1], device = device),              # Middle of doublet 1                                                     # Middle of doublet 1
    do.XYPolynomial(d_R, d_F - d_length/2 + d_x3-d_x0, J=2, ai=d_coefs[2], device = device),              # End of doublet 1                                                     # End of doublet 1
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2, J=1, ai=d_coefs[3][:3], device = device),                                                                   # Prism 1
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2 + d_R*np.tan(angle*np.pi/180), J=1, ai=d_coefs[4][:3], device = device),                                       # Prism 2
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim - prism_length/2 + (d_H+d_R)*np.tan(angle*np.pi/180), J=1, ai=d_coefs[5][:3], device = device),                                   # Prism 3
    do.XYPolynomial(d_R, 2*d_F + d_offset_collim + prism_length/2, J=1, ai=d_coefs[6][:3], device = device),                                     # End of prism
    do.XYPolynomial(d_R, 3*F + 2*d_offset_collim + d_length/2 - d_x3-d_x0, J=2, ai=d_coefs[7], device = device),  # Start of doublet 2                                                    # Start of doublet 2
    do.XYPolynomial(d_R, 3*F + 2*d_offset_collim + d_length/2 - d_x2-d_x0, J=2, ai=d_coefs[8], device = device),  # Middle of doublet 2                                                    # Middle of doublet 2
    do.XYPolynomial(d_R, 3*F + 2*d_offset_collim + d_length/2 - d_x1-d_x0, J=2, ai=d_coefs[9], device = device),  # End of doublet 2                                                    # End of doublet 2
]


d_materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('sf5'),
    do.Material('air'),
    do.Material('N-SK2'),
    do.Material('N-SF4'),
    do.Material('N-SK2'),
    do.Material('air'),
    do.Material('sf5'),
    do.Material('N-BK7'),
    do.Material('air')
]

d_lens = do.Lensgroup(device=device)
d_lens.load(d_surfaces, d_materials)
d_lens.d_sensor = 4*d_F + 2
d_lens.r_last = d_R

d_lens.film_size = [131, 131] # [pixels]
d_lens.pixel_size = 80.0e-3 # [mm]

################################################################################################################################################

simple_coefs = torch.tensor([[0, 0, 0, 0, 0, 0], 
            [x0-e, 0, 0, -(R+e-x0)/(x0*x0), 0, -(R+e-x0)/(x0*x0)]
]).float()

simple_surfaces = [
    do.XYPolynomial(R, 2*F - te, J=1, ai=coefs[0][:3], device = device),                    # Start of lens 1
    do.XYPolynomial(R, 2*F - x0+2*e, J=2, ai=coefs[1], device = device),                    # End of lens 1
]

simple_materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air'),
]

simple_lens = do.Lensgroup(device=device)
simple_lens.load(simple_surfaces, simple_materials)
simple_lens.d_sensor = 4*F - 14 #+tc # distance to sensor from 0 (in mm?)
simple_lens.r_last = R             # radius of sensor (in mm?) in plot, changed in prepare_mts

simple_lens.film_size = [131, 131] # [pixels]
simple_lens.pixel_size = 80.0e-3*2.5   # [mm]

################################################################################################################################################

doublegauss_lens = do.Lensgroup(device=device)
doublegauss_lens.load_file(Path('./lenses/DoubleGauss/US02532751-1.txt'))

def test_propag():
    # set a rendering image sensor, and call prepare_mts to prepare the lensgroup for rendering
    lens.prepare_mts(lens.pixel_size, lens.film_size)
    """ for surface in lens.surfaces:
        print(surface.d) """
    #lens.plot_setup2D()

    # create a dummy screen
    z0 = 131 #112 # [mm]
    z0 = 1000
    z0 = d+F+x2+F
    z0 = 4*F
    #z0 = F
    #z0 = 0

    pixelsize = 80e-3 #70e-3 # [mm]
    pixelsize = 80e-3

    nb_wavelengths = 3
    size_pattern = (131,131)
    texture = 255*np.random.choice([0, 1], size=size_pattern + (nb_wavelengths,), p=[1 - 0.5, 0.5]).astype(np.float32)
    texture = np.zeros(size_pattern + (nb_wavelengths,)).astype(np.float32)
    texture[10:-20, size_pattern[1]//2, :] = 255
    texture[size_pattern[0]//2, :, :] = 255
    texture[size_pattern[0]//2-15:size_pattern[0]//2+15, size_pattern[1]//2+40, :] = 255
    #texture[:size_pattern[0]//2, :size_pattern[1]//2, :] = 255
    #texture[size_pattern[0]//2-10:size_pattern[0]//2+10, size_pattern[1]//2-10:size_pattern[1]//2+10, :] = 255

    #texture = mpimg.imread('./images/squirrel.jpg')[500:500+131,500:500+131,:]

    plt.plot()
    plt.imshow(texture)
    plt.show()
    #texture = 255*np.ones((131, 131, nb_wavelengths)).astype(np.float32)

    texture_torch = torch.Tensor(texture).float().to(device=device)
    texturesize = np.array(texture.shape[0:2])
    screen = do.Screen(
        do.Transformation(np.eye(3), np.array([0, 0, z0])),
        texturesize * pixelsize, texture_torch, device=device
    )

    # helper function
    def render_single(wavelength, screen):
        valid, ray_new = lens.sample_ray_sensor(wavelength)
        uv, valid_screen = screen.intersect(ray_new)[1:]
        mask = valid & valid_screen
        I = screen.shading(uv, mask)
        return I, mask

    # sample wavelengths in [nm]
    wavelengths = [656.2725, 587.5618, 486.1327]
    wavelengths = wavelengths[:nb_wavelengths]

    # render
    ray_counts_per_pixel = 20
    time_start = time.time()
    Is = []
    for wavelength_id, wavelength in enumerate(wavelengths):
        screen.update_texture(texture_torch[..., wavelength_id])

        # multi-pass rendering by sampling the aperture
        I = 0
        M = 0
        for i in tqdm(range(ray_counts_per_pixel)):
            I_current, mask = render_single(wavelength, screen)
            I = I + I_current
            M = M + mask
        I = I / (M + 1e-10)
        
        # reshape data to a 2D image
        I = I.reshape(*np.flip(np.asarray(lens.film_size))).permute(1,0)
        #I = torch.flip(I.reshape(*np.flip(np.asarray(lens.film_size))).permute(1,0), dims = [0,1])
        Is.append(I.cpu())
    print(time.time()-time_start)
    # show image
    I_rendered = torch.stack(Is, axis=-1).numpy().astype(np.uint8)
    plt.imshow(I_rendered)
    plt.show()
    plt.imshow(np.sum(I_rendered, axis=-1))
    plt.show()
    plt.imsave(save_dir + 'I_rendered.png', I_rendered)

def test_psf():
    # sensor area
    pixel_size = 6.45e-3 # [mm]
    film_size = torch.tensor([1200, 1600], device=device)

    #pixel_size = lens.pixel_size
    pixel_size = 80.0e-3/2
    #film_size = 10*torch.tensor(lens.film_size, device=device)

    R_square = film_size * pixel_size

    # generate array of rays
    wavelength = 600 # [nm]
    R = 12.7 # [mm]
    #lens.plot_setup2D()

    def render_psf(I, p):
        # compute shifts and do linear interpolation
        uv = (p + R_square/2) / pixel_size
        index_l = torch.vstack((
            torch.clamp(torch.floor(uv[:,0]).long(), min=0, max=film_size[0]),
            torch.clamp(torch.floor(uv[:,1]).long(), min=0, max=film_size[1]))
        ).T
        index_r = torch.vstack((
            torch.clamp(index_l[:,0] + 1, min=0, max=film_size[0]),
            torch.clamp(index_l[:,1] + 1, min=0, max=film_size[1]))
        ).T
        w_r = torch.clamp(uv - index_l, min=0, max=1)
        w_l = 1.0 - w_r
        del uv

        # compute image
        I = torch.index_put(I, (index_l[...,0],index_l[...,1]), w_l[...,0]*w_l[...,1], accumulate=True)
        I = torch.index_put(I, (index_r[...,0],index_l[...,1]), w_r[...,0]*w_l[...,1], accumulate=True)
        I = torch.index_put(I, (index_l[...,0],index_r[...,1]), w_l[...,0]*w_r[...,1], accumulate=True)
        I = torch.index_put(I, (index_r[...,0],index_r[...,1]), w_r[...,0]*w_r[...,1], accumulate=True)
        return I

    def generate_surface_samples(M):
        Dx = np.random.rand(M,M)
        Dy = np.random.rand(M,M)
        [px, py] = do.Sampler().concentric_sample_disk(Dx, Dy)
        return np.stack((px.flatten(), py.flatten()), axis=1)

    def sample_ray(o_obj, M):
        p_aperture_2d = R * generate_surface_samples(M)
        #print(p_aperture_2d.shape)
        #plt.plot(p_aperture_2d[:,0], p_aperture_2d[:,1], 'o')
        #plt.show()
        N = p_aperture_2d.shape[0]
        p_aperture = np.hstack((p_aperture_2d, np.zeros((N,1)))).reshape((N,3))
        o = np.ones(N)[:, None] * o_obj[None, :]
        
        o = o.astype(np.float32)
        p_aperture = p_aperture.astype(np.float32)
        
        d = do.normalize(torch.from_numpy(p_aperture - o))
        
        o = torch.from_numpy(o).to(lens.device)
        d = d.to(lens.device)
        
        return do.Ray(o, d, wavelength, device=lens.device)

    def render(o_obj, M, rep_count):
        I = torch.zeros(*film_size, device=device)
        for i in range(rep_count):
            rays = sample_ray(o_obj, M)
            ps = lens.trace_to_sensor(rays, ignore_invalid=True)
            I = render_psf(I, ps[..., :2])
        return I / rep_count

    # PSF rendering parameters
    x_max_halfangle = 10 # [deg]
    y_max_halfangle = 7.5 # [deg]
    Nx = 2 * 8 + 1
    Ny = 2 * 6 + 1

    # sampling parameters
    M = 50
    rep_count = 1

    def render_at_depth(z):
        x_halfmax = np.abs(z) * np.tan(np.deg2rad(x_max_halfangle))
        y_halfmax = np.abs(z) * np.tan(np.deg2rad(y_max_halfangle))

        I_psf_all = torch.zeros(*film_size, device=device)
        for x in tqdm(np.linspace(-x_halfmax, x_halfmax, Nx)):
            for y in np.linspace(-y_halfmax, y_halfmax, Ny):
                o_obj = np.array([y, x, z])
                I_psf = render(o_obj, M, rep_count)
                I_psf_all = I_psf_all + I_psf
        return I_psf_all

    # render PSF at different depths
    zs = [-1e4, -7e3, -5e3, -3e3, -2e3, -1.5e3, -1e3]
    zs = [-5e4, -1e4, -1e3, -2e3, -1.5e3, -1e3]
    #zs = [zs[0]]
    savedir = Path('./rendered_psfs')
    savedir.mkdir(exist_ok=True, parents=True)
    I_psfs = []
    for z in zs:
        I_psf = render_at_depth(z)
        I_psf = I_psf.cpu().numpy()
        plt.imsave(str(savedir / 'I_psf_z={z}_w={w}.svg'.format(z=z, w=wavelength)), np.uint8(255 * I_psf / I_psf.max()), cmap='hot', format='svg')
        I_psfs.append(I_psf)

def sample_rays_pos(wavelength, angles, x_pos, y_pos, z_pos):
    angles_rad = torch.tensor(np.radians(np.asarray(angles)))
    angles_phi = angles_rad[:,0]
    angles_psi = angles_rad[:,1]
    
    pos = torch.tensor([x_pos, y_pos, z_pos]).repeat(angles_rad.shape[0], 1).float()

    o = pos
    d = torch.stack((
        torch.sin(angles_phi),
        torch.sin(angles_psi),
        torch.cos(angles_phi)*torch.cos(angles_psi)), axis=-1
    ).float()
    
    """ d = torch.stack((
        torch.sin(angles_psi)*torch.cos(angles_phi),
        torch.sin(angles_psi)*torch.sin(angles_phi),
        torch.cos(angles_psi)), axis=-1
    ).float() """

    d = d/torch.norm(d, p=2, dim=1)[:, None]

    return do.Ray(o, d, wavelength, device=lens.device)

def sanity_check(M=1, R=None, lens=lens, angles=[0], x_pos=0, y_pos=0, z_pos=0, wavelength = 550):
    # sample wavelengths in [nm]
    wavelength = torch.Tensor([wavelength]).float().to(device)

    colors_list = 'bgry'
    views = np.array([0])
    ax, fig = lens.plot_setup2D_with_trace(views, wavelength, M=4)
    ax.axis('off')
    ax.set_title('Sanity Check Setup 2D')
    fig.savefig('sanity_check_setup.pdf')

    # spot diagrams
    spot_rmss = []
    #ray = lens.sample_ray(wavelengths, view=view, M=M, R=R, sampling='grid', entrance_pupil=True)
    ray = sample_rays_pos(wavelength, angles, x_pos, y_pos, z_pos)
    print(ray)
    ps = lens.trace_to_sensor(ray, ignore_invalid=True)
    normalize = True
    if normalize:
        lim = 1
    else:
        lim = 14
    lens.spot_diagram(
        ps[...,:2], show=True, xlims=[-lim, lim], ylims=[-lim, lim], color=colors_list[0]+'.',
        savepath='sanity_check_field.png', normalize = normalize
    )

    spot_rmss.append(lens.rms(ps))

    plt.show()

if __name__ == '__main__':
    #sample_rays_pos(torch.tensor([550]), [50, 60], 1, 1)
    #sanity_check(M=2, R=1)
    N = 10000
    max_angle = 5
    x_pos, y_pos, z_pos = 0, 0, 10
    angles_phi = 2*max_angle*np.random.rand(N) - max_angle
    angles_psi = 2*max_angle*np.random.rand(N) - max_angle
    angles = np.stack((angles_phi, angles_psi), axis=-1)
    sanity_check(lens = d_lens, angles = angles, x_pos = x_pos, y_pos = y_pos, z_pos = z_pos, wavelength = 550)

    #test_propag()