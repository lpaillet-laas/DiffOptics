import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../")
import diffoptics as do

import time

def create_lensgroup(surfaces, materials, device, d_sensor, r_last, film_size=None, pixel_size=None,
                     theta_x=0., theta_y=0., theta_z=0., origin=np.zeros(3), shift=np.zeros(3)):
    """
    Create a lens group object based on the given parameters. Distances in the surfaces are relative to the last surface.

    Args:
        surfaces (list[dict]): List of dictionaries representing the surfaces. Each dictionary should contain the following keys:
            - type (str): Type of the surface. Can be 'Aspheric', 'XYPolynomial', or 'BSpline'.
            - R (float): Radius of the surface.
            - d (float): Distance from the last surface.
            - params (dict): Parameters of the surface. The allowed fields depend on the surface type:
                - For 'Aspheric': 'c' (float), 'k' (float), 'ai' (float or None), 'is_square' (bool).
                - For 'XYPolynomial': 'J' (float), 'ai' (float or None), 'b' (float or None), 'is_square' (bool).
                - For 'BSpline': 'size' (tuple[int, int]) necessary, 'px' (int), 'py' (int), 'tx' (float or None), 'ty' (float or None), 'c' (float or None), 'is_square' (bool).
        materials (list[str]): List of the names of the desired materials.
        device (str): Device to be used for the computation.
        d_sensor (float): Distance from the origin to the sensor.
        r_last (float): Radius of the last surface.
        film_size (tuple[int, int], optional): Film size of the sensor in x, y coordinates. Defaults to None.
        pixel_size (float, optional): Pixel size of the sensor. Defaults to None.
        theta_x (float, optional): Rotation angle (in degrees) along the x axis for the entire lens group. Defaults to 0.
        theta_y (float, optional): Rotation angle (in degrees) along the y axis for the entire lens group. Defaults to 0.
        theta_z (float, optional): Rotation angle (in degrees) along the z axis for the entire lens group. Defaults to 0.
        origin (tuple[float, float, float], optional): Origin position in x, y, z coordinates. Defaults to (0, 0, 0).
        shift (tuple[float, float, float], optional): Shift of the lens group relative to the origin, in x, y, z coordinates. Defaults to (0, 0, 0).

    Returns:
        Lensgroup: Lens group object representing the created lens group.
    """
    surfaces_processed = []
    accumulated_d = 0
    for surface in surfaces:
        if surface['type'] == 'Aspheric':
            arguments = ['c', 'k', 'ai', 'is_square']
            for argument in arguments:
                if argument not in surface['params']:
                    if argument in ['c', 'k']:
                        surface['params'][argument] = 0.
                    elif argument == 'ai':
                        surface['params'][argument] = None
                    elif argument == 'is_square':
                        surface['params'][argument] = False

            surfaces_processed.append(do.Aspheric(
                surface['R'], surface['d']+accumulated_d, c=surface['params']['c'], ai=surface['params']['ai'], is_square=surface['params']['is_square'], device=device
            ))
        elif surface['type'] == 'XYPolynomial':
            arguments = ['J', 'ai', 'b', 'is_square']
            for argument in arguments:
                if argument not in surface['params']:
                    if argument == 'J':
                        surface['params'][argument] = 0.
                    elif argument in ['ai', 'b']:
                        surface['params'][argument] = None
                    elif argument == 'is_square':
                        surface['params'][argument] = False
            surfaces_processed.append(do.XYPolynomial(
                surface['R'], surface['d']+accumulated_d, J=surface['params']['J'], ai=surface['params']['ai'], b=surface['params']['b'], is_square=surface['params']['is_square'], device=device
            ))
        elif surface['type'] == 'BSpline':
            arguments = ['size', 'px', 'py', 'tx', 'ty', 'c', 'is_square']
            for argument in arguments:
                if argument not in surface['params']:
                    if argument =='size':
                        raise AttributeError('size must be provided for BSpline surface.')
                    elif argument in ['px', 'py']:
                        surface['params'][argument] = 3
                    elif argument in ['tx', 'ty', 'c']:
                        surface['params'][argument] = None
                    elif argument == 'is_square':
                        surface['params'][argument] = False
            surfaces_processed.append(do.BSpline(
                surface['R'], surface['d']+accumulated_d, size=surface['params']['size'], px=surface['params']['px'], py=surface['params']['py'], tx=surface['params']['tx'], ty=surface['params']['ty'], c=surface['params']['c'], is_square=surface['params']['is_square'], device=device
            ))
        accumulated_d += surface['d']
    
    materials_processed = []
    for material in materials:
        materials_processed.append(do.Material(material))

    lens = do.Lensgroup(device=device, theta_x=theta_x, theta_y=theta_y, theta_z=theta_z, origin=origin, shift=shift)
    lens.load(surfaces_processed, materials_processed)
    lens.d_sensor = d_sensor
    lens.r_last = r_last
    lens.film_size = film_size
    lens.pixel_size = pixel_size

    return lens

def trace_all(lenses, R=12.5, M=5, wavelength=550):
    """
    Trace rays through a series of lenses and calculate the ray positions and optical path lengths (oss).

    Args:
        lenses (list): A list of lens objects representing the lenses in the system.
        R (float): The radius of the incoming beam of rays.
        M (int): The number of rays to sample.
        wavelength (float): The wavelength of the light.

    Returns:
        tuple: A tuple containing the ray positions (ps) and optical path lengths (oss).
    """

    # Generate initial ray positions and optical path lengths
    ray_mid = lenses[0].sample_ray_2D(R, wavelength, M=M)
    oss = [0 for i in range(len(lenses))]
    
    # Trace rays through each lens in the system
    for i, lens in enumerate(lenses[:-1]):
        ray_mid, valid, oss_temp = lens.trace_r(ray_mid)
        oss[i] = oss_temp
    
    # Trace rays to the sensor
    ps, oss_last = lenses[-1].trace_to_sensor_r(ray_mid)
    oss[-1] = oss_last
    
    # Return the ray positions and optical path lengths
    return ps[...,:2], oss

def combined_plot_setup(lenses, with_sensor=False):
    """
    Plot the setup of multiple lenses in a combined figure.

    Args:
        lenses (list): A list of Lens objects representing the lenses in the setup.
        with_sensor (bool, optional): Whether to include the sensor in the plot. Defaults to False.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure object.
        ax (matplotlib.axes.Axes): The generated axes object.
    """
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the setup of each lens in 2D
    for i, lens in enumerate(lenses[:-1]):
        lens.plot_setup2D(ax=ax, fig=fig, with_sensor=with_sensor, show=False)
    
    # Plot the setup of the last lens with the sensor
    lenses[-1].plot_setup2D(ax=ax, fig=fig, with_sensor=with_sensor, show=True)
    
    # Return the figure and axes objects
    return fig, ax

def prepare_mts(lens, pixel_size, film_size, R=np.eye(3), t=np.zeros(3), start_distance=0.0):
    """
    Prepares the lens for multi-surface tracing (MTS) by setting the necessary parameters. This function should be called before rendering the lens.
    It is based on the original prepare_mts function in the diffoptics library, with the option to specify the starting distance for the lens surfaces to allow for smoother rendering with several lens groups.

    Args:
        lens (Lens): The lens object to prepare for MTS.
        pixel_size (float): The size of a pixel in millimeters.
        film_size (length 2 int array): The size of the film in pixels.
        R (np.ndarray, optional): The rotation matrix representing the transformation of the lensgroup. Defaults to np.eye(3).
        t (np.ndarray, optional): The translation vector representing the transformation of the lensgroup. Defaults to np.zeros(3).
        start_distance (float, optional): The starting distance for the lens surfaces. Defaults to 0.0.
    """
    
    if lens.mts_prepared:
        print('MTS already prepared for this lensgroup.')
        return
        
    # sensor parameters
    lens.pixel_size = pixel_size # [mm]
    lens.film_size = film_size # [pixel]

    # rendering parameters
    lens.mts_Rt = do.Transformation(R, t) # transformation of the lensgroup
    lens.mts_Rt.to(lens.device)

    # for visualization
    lens.r_last = lens.pixel_size * max(lens.film_size) / 2

    # treat the lenspart as a camera; append one more surface to it
    lens.surfaces.append(do.Aspheric(lens.r_last, lens.d_sensor, 0.0))

    # reverse surfaces
    d_total = lens.surfaces[-1].d
    for i in range(len(lens.surfaces)):
        lens.surfaces[i].d = d_total - lens.surfaces[i].d + start_distance
        lens.surfaces[i].reverse()
    lens.surfaces.reverse()
    lens.surfaces.pop(0) # remove sensor plane

    # reverse materials
    lens.materials.reverse()

    # aperture plane
    lens.aperture_radius = lens.surfaces[0].r
    lens.aperture_distance = lens.surfaces[0].d
    lens.mts_prepared = True
    lens.d_sensor = 0

def render_single_back(lenses, wavelength, screen, aperture_reduction = 1):
    """
    Renders back propagation of single ray through a series of lenses onto a screen.

    Args:
        lenses (list): A list of lens objects representing the lenses in the system.
        wavelength (float): The wavelength of the light.
        screen (object): The screen object onto which the light is projected.

    Returns:
        tuple: A tuple containing the intensity values (I) and the mask indicating valid pixels on the screen.
    """
    # Sample rays from the sensor
    valid_1, ray_mid = lenses[-1].sample_ray_sensor(wavelength, aperture_reduction = aperture_reduction)
    ray_mid.o = ray_mid.o[valid_1, :]
    ray_mid.d = ray_mid.d[valid_1, :]
    
    # Trace rays through each lens in the system
    for lens in lenses[1:-1]:
        valid, ray_mid = lens._trace(ray_mid)
        ray_mid.o = ray_mid.o[valid, :]
        ray_mid.d = ray_mid.d[valid, :]
        ray_mid = lens.mts_Rt.transform_ray(ray_mid)
    
    # Trace rays to the first lens
    valid_last, ray_last = lenses[0]._trace(ray_mid)
    ray_last = lenses[0].mts_Rt.transform_ray(ray_last)
    
    # Intersect rays with the screen
    uv, valid_screen = screen.intersect(ray_last)[1:]
    
    # Apply mask to filter out invalid rays
    mask = valid_last & valid_screen
    
    # Calculate intensity values on the screen
    I = screen.shading(uv, mask)
    
    return I, mask

def plot_setup_basic_rays(lenses):
    """
    Plot the setup with a beam of rays.

    Args:
        lenses (list): A list of lenses in the setup.

    Returns:
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
    """
    ps, oss = trace_all(lenses)
    return plot_setup_with_rays(lenses, oss)

def plot_setup_with_rays(lenses, oss):
    """
    Plots the setup with rays for a given list of lenses and optical systems.

    Args:
        lenses (list): A list of lenses.
        oss (list): A list of optical system records.

    Returns:
        ax (matplotlib.axes.Axes): The matplotlib axes object.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
    """
    
    # If there is only one lens, plot the raytraces with the sensor
    if len(lenses)==1:
        ax, fig = lenses[0].plot_raytraces(oss[0], show=True, with_sensor=True)
        return ax, fig
    
    # Plot the raytraces for the first lens without the sensor
    ax, fig = lenses[0].plot_raytraces(oss[0], color='b-', show=False, with_sensor=False)
    
    # Plot the raytraces for the intermediate lenses without the sensor
    for i, lens in enumerate(lenses[1:-1]):
        ax, fig = lens.plot_raytraces(oss[i+1], ax=ax, fig=fig, color='b-', show=False, with_sensor=False)
    
    # Plot the raytraces for the last lens with the sensor
    ax, fig = lenses[-1].plot_raytraces(oss[-1], ax=ax, fig=fig, color='b-', show=True, with_sensor=True)
    
    return ax, fig



def propagate(lenses, texture = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327], z0=0, offsets=None,
                aperture_reduction = 1, save_dir = None):
    
    """
    Perform ray tracing simulation for propagating light through a lens system. Renders the texture on a screen

    Args:
        lenses (list): List of lens objects representing the lens system.
        texture (ndarray, optional): Texture pattern used for rendering. Defaults to None.
        nb_rays (int, optional): Number of rays to be traced per pixel. Defaults to 20.
        wavelengths (list, optional): List of wavelengths to be simulated. Defaults to [656.2725, 587.5618, 486.1327] (RGB).
        z0 (float, optional): Initial z-coordinate of the screen. Defaults to 0.
        offsets (list, optional): List of offsets for each lens in the system. Defaults to None.
        save_dir (str, optional): Directory to save the rendered image. Defaults to None.

    Returns:
        I_rendered (ndarray): The rendered image.
    """
    if offsets is None:
        offsets = [0 for i in range(len(lenses))]

    # set a rendering image sensor, and call prepare_mts to prepare the lensgroup for rendering
    for i, lens in enumerate(lenses[::-1]):
        lens_mts_R, lens_mts_t = lens._compute_transformation().R, lens._compute_transformation().t
        if i > 0:
            prepare_mts(lens, lens.pixel_size, lens.film_size, start_distance = lenses[::-1][i-1].surfaces[-1].d + offsets[-i-1], R=lens_mts_R, t=lens_mts_t)
        else:
            prepare_mts(lens, lens.pixel_size, lens.film_size, start_distance = offsets[-1], R=lens_mts_R, t=lens_mts_t)    
    
    combined_plot_setup(lenses, with_sensor=True)

    # create a dummy screen
    pixelsize = lenses[0].pixel_size # [mm]

    nb_wavelengths = len(wavelengths)
    if texture is None:
        size_pattern = tuple(lenses[0].film_size)
        texture = np.zeros(size_pattern + (nb_wavelengths,)).astype(np.float32)
        texture[int(0.1*size_pattern[0]):-int(0.3*size_pattern[0]), size_pattern[1]//2 + 0*4, :] = 1                # Big vertical
        texture[size_pattern[0]//2, int(0.2*size_pattern[1]):int(0.8*size_pattern[1]), :] = 1                 # Horizontal
        texture[int(0.4*size_pattern[0]):int(0.6*size_pattern[0]), int(0.7*size_pattern[1]) + 0*4, :] = 1           # Small vertical

    if nb_wavelengths == 3:
        plt.figure()
        plt.plot()
        plt.imshow(texture)

    texture_torch = torch.Tensor(texture).float().to(device=lenses[0].device)
    texture_torch = torch.permute(texture_torch, (1,0,2)) # Permute
    texture_torch = texture_torch.flip(dims=[0]) # Flip
    texturesize = np.array(texture.shape[0:2])
    screen = do.Screen(
        do.Transformation(np.eye(3), np.array([0, 0, z0])),
        texturesize * pixelsize, texture_torch, device=lenses[0].device
    )
    print("Texture nonzero: ", texture_torch.count_nonzero())

    # render
    ray_counts_per_pixel = nb_rays
    time_start = time.time()
    Is = []
    for wavelength_id, wavelength in enumerate(wavelengths):
        screen.update_texture(texture_torch[..., wavelength_id])

        # multi-pass rendering by sampling the aperture
        I = 0
        M = 0
        for i in tqdm(range(ray_counts_per_pixel)):
            I_current, mask = render_single_back(lenses, wavelength, screen, aperture_reduction=aperture_reduction)
            I = I + I_current
            M = M + mask
        I = I / (M + 1e-10)
        # reshape data to a 2D image
        #I = I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0)
        print(f"Image {wavelength_id} nonzero count: {I.count_nonzero()}")
        #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0), dims = [0])
        #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims = [0])
        #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims=[0, 1])
        I = I.reshape(*np.flip(np.asarray(lenses[0].film_size))).flip(1) # Flip
        Is.append(I.cpu())
    # show image
    I_rendered = torch.stack(Is, axis=-1).numpy()#.astype(np.uint8)
    print(f"Elapsed rendering time: {time.time()-time_start:.3f}s")

    if nb_wavelengths==3:
        plt.imshow(np.flip(I_rendered/I_rendered.max(axis=(0,1))[np.newaxis, np.newaxis, :], axis=2))
        plt.show()
    ax, fig = plt.subplots((nb_wavelengths+2)// 3, 3, figsize=(15, 5))
    ax.suptitle("Rendered with dO")
    for i in range(nb_wavelengths):
        fig.ravel()[i].set_title("Wavelength: " + str(float(wavelengths[i])) + " nm")
        if nb_wavelengths > 3:
            fig[i//3, i % 3].imshow(I_rendered[:,:,i])
        else:
            fig[i].imshow(I_rendered[:,:,i])
        if save_dir is not None:
            plt.imsave(os.path.join(save_dir, f"rendered_{wavelengths[i]}.png"), I_rendered[:,:,i])
    plt.show()
    plt.imshow(np.sum(I_rendered, axis=-1))
    plt.show()
    if nb_wavelengths==3 and save_dir is not None:
        plt.imsave(os.path.join(save_dir, "rendered_rgb.png"), I_rendered/I_rendered.max(axis=(0,1))[np.newaxis, np.newaxis, :])

    return I_rendered


def sample_rays_pos(wavelength, angles, x_pos, y_pos, z_pos, device, d = None):
    """
    Samples rays with given wavelength, angles, and positions.

    Args:
        wavelength (float): The wavelength of the rays.
        angles (list): A list of angles in degrees, where each angle is represented as a tuple (phi, psi).
        x_pos (float): The x-coordinate of the position.
        y_pos (float): The y-coordinate of the position.
        z_pos (float): The z-coordinate of the position.
        device (torch.device): The device on which to create the rays.

    Returns:
        ray (do.Ray): A Ray object representing the sampled rays.
    """
    
    if d is None:
        # Convert angles from degrees to radians
        angles_rad = torch.tensor(np.radians(np.asarray(angles)))
        angles_phi = angles_rad[:,0]
        angles_psi = angles_rad[:,1]
        
        # Create position tensor
        pos = torch.tensor([x_pos, y_pos, z_pos]).repeat(angles_rad.shape[0], 1).float()

        # Calculate ray origin and direction
        o = pos
        d = torch.stack((
            torch.sin(angles_phi),
            torch.sin(angles_psi),
            torch.cos(angles_phi)*torch.cos(angles_psi)), axis=-1
        ).float()
    else:
        pos = torch.tensor([x_pos, y_pos, z_pos]).repeat(d.shape[0], 1).float()

        o = pos
        d = torch.Tensor(d).float() if type(d) == np.ndarray else d.float()

    """ d = torch.stack((
        torch.sin(angles_phi),
        0*torch.sin(angles_psi),
        torch.cos(angles_phi)), axis=-1
    ).float() """
    
    """ d = torch.stack((
        torch.sin(angles_psi)*torch.cos(angles_phi),
        torch.sin(angles_psi)*torch.sin(angles_phi),
        torch.cos(angles_psi),), axis=-1
    ).float() """

    d = d/torch.norm(d, p=2, dim=1)[:, None]

    return do.Ray(o, d, wavelength, device=device)

def trace_psf_from_point_source(lenses=None, angles=[[0, 0]], x_pos=0, y_pos=0, z_pos=0, wavelength = 550, normalize = True,
                 show_rays = True, save_dir = None, d = None, ignore_invalid = False, show_res = True):
    """
    Traces the point spread function (PSF) from a point source through a series of lenses.

    Args:
        lenses (list): List of lenses through which the rays will be traced
        angles (list): angles (list): A list of angles in degrees, where each angle is represented as a tuple (phi, psi) (default: [[0,0]]).
        x_pos (int): X position of the point source (default: 0)
        y_pos (int): Y position of the point source (default: 0)
        z_pos (int): Z position of the point source (default: 0)
        wavelength (int): Wavelength of the rays in nanometers (default: 550)
        normalize (bool): Flag indicating whether to normalize the PSF (default: True)
        show (bool): Flag indicating whether to show the plots (default: True)

    Returns:
        ps (array): Array containing the x and y coordinates of the PSF
    """
    
    if lenses is None:
        raise AttributeError('lenses must be provided')
    
    # sample wavelengths in [nm]
    wavelength = torch.Tensor([wavelength]).float().to(lenses[0].device)

    colors_list = 'bgry'
    """ views = np.array([0])
    ax, fig = lens.plot_setup2D_with_trace(views, wavelength, M=4)
    ax.axis('off')
    ax.set_title('Sanity Check Setup 2D')
    fig.savefig('sanity_check_setup.pdf') """

    ray = sample_rays_pos(wavelength, angles, x_pos, y_pos, z_pos, lenses[0].device, d = d)
    
    pos = ray(torch.tensor([70.120]))
    plt.scatter(pos[:,1], pos[:,0])
    plt.axis('scaled')
    plt.title('Entry of first lens')
    #plt.show()
    
    oss = [0 for i in range(len(lenses))]
    for i, lens in enumerate(lenses[:-1]):
        if show_rays:
            ray, valid, oss_mid = lens.trace_r(ray)
            oss[i] = oss_mid
        else:
            ray, valid = lens.trace(ray)
        if not ignore_invalid:
            ray.o = ray.o[valid, :]
            ray.d = ray.d[valid, :]

    pos = ray(torch.tensor([np.cos(-9.088)*75.0]))
    plt.figure()
    plt.scatter(pos[:,1], pos[:,0])
    plt.axis('scaled')
    plt.title('Entry of second lens')
    
    #print("Mean angle out of first n-1 groups: ", torch.mean(torch.atan2(ray.d[:,0], ray.d[:, 2])*180/np.pi))

    if show_rays:
        ps, oss_final = lenses[-1].trace_to_sensor_r(ray, ignore_invalid=True)
        oss[-1] = oss_final
        ax, fig = plot_setup_with_rays(lenses, oss)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "setup_with_rays.svg"), format="svg")
    else:
        ps = lenses[-1].trace_to_sensor(ray, ignore_invalid=True)
        
    if normalize:
        if type(normalize) == bool:
            lim = 0.04
        else:
            lim = normalize        
    else:
        lim = 14
    lenses[-1].spot_diagram(
        ps[...,:2], show=show_res, xlims=[-lim, lim], ylims=[-lim, lim], color=colors_list[0]+'.',
        savepath='sanity_check_field.png', normalize = normalize
    )

    if not show_res:
        plt.close()
    return ps[...,:2]

def plot_spot_diagram(lenses, wavelength, nb_elems, nb_pixels, size_pixel, normalize=False, show=True,
                    save_dir = "./", device='cpu'):
    """
    Generates a spot diagram for a given set of lenses.

    Args:
        lenses (list): A list of lenses.
        wavelength (float): The wavelength of the light.
        nb_elems (int): The number of elements in the grid.
        max_val (float): The maximum value for the grid.
        normalize (bool, optional): Whether to normalize the spot diagram. Defaults to False.
        show (bool, optional): Whether to display the spot diagram. Defaults to True.
        save_dir (str, optional): The directory to save the spot diagram. Defaults to "./".

    Returns:
        numpy.ndarray: The spot diagram coordinates.
    """
    ray = lenses[0].sample_ray(wavelength, M=nb_elems, R=nb_pixels*size_pixel/2, sampling='grid')

    # CASSI mesh 
    x, y = torch.meshgrid(
        torch.arange(-(nb_pixels-1)*size_pixel/2, (nb_pixels+1)*size_pixel/2, size_pixel, device=device),
        torch.arange(-(nb_pixels-1)*size_pixel/2, (nb_pixels+1)*size_pixel/2, size_pixel, device=device),
        indexing='xy' # ij in original code
    )

    np.save(save_dir + "grid.npy", np.stack((x.cpu().detach().numpy().flatten(),y.cpu().detach().numpy().flatten()), axis=-1))

    o = torch.stack((x,y,torch.zeros_like(x, device=device)), axis=2)
    ray.o = o
    for i, lens in enumerate(lenses[:-1]):
        ray, valid = lens.trace(ray)
    ps = lenses[-1].trace_to_sensor(ray)
    lenses[-1].spot_diagram(ps, xlims=[-nb_pixels*size_pixel/2*1.5, nb_pixels*size_pixel/2*1.5], ylims=[-nb_pixels*size_pixel/2*1.5, nb_pixels*size_pixel/2*1.5], savepath=save_dir + "spotdiagram_doublet" + ".png", normalize=normalize, show=show)
    ps = ps.cpu().detach().numpy()
    return ps[...,:2]
    
