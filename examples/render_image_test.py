import numpy as np
#import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.image as mpimg
import time

import sys
sys.path.append("../")
import diffoptics as do

# initialize a lens
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lens = do.Lensgroup(device=device)

# load optics
lens.load_file(Path('./lenses/DoubleGauss/US02532751-1.txt'))

# set coded aperture pixel size and nb of pixels
pixel_size = 80e-3 # [mm]
film_size = [131, 131]

# set a rendering image sensor, and call prepare_mts to prepare the lensgroup for rendering
lens.prepare_mts(pixel_size, film_size)
# lens.plot_setup2D()

# create a dummy screen
z0 = 131 #112 # [mm]
pixelsize = 80e-3 #70e-3 # [mm]
texture = np.ones((131, 131, 55)).astype(np.float32)

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
wavelengths = np.linspace(450, 650, 55)

# render
ray_counts_per_pixel = 1
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
    I = I.reshape(*np.flip(np.asarray(film_size))).permute(1,0)
    Is.append(I.cpu())

# show image
I_rendered = torch.stack(Is, axis=-1).numpy().astype(np.uint8)
plt.imshow(I_rendered[...,0])
plt.show()
plt.imshow(np.sum(I_rendered, axis=-1))
plt.show()
#plt.imsave('I_rendered.png', I_rendered)

time_start = time.time()
for i in range(len(wavelengths)):
    #valid, ray_new = lens.sample_ray_sensor(wavelengths[i])
    ray_new = lens.sample_ray(wavelengths[i], view=0, M=131, R=130*40*1e-3, sampling='grid', entrance_pupil=False)
    if i>0:
        x_y_values = torch.cat((x_y_values, ray_new(torch.tensor([75000*1e-6]))[...,0:2][..., None]), dim=-1)
    else:
        x_y_values = ray_new(torch.tensor([75000*1e-6]))[...,0:2][...,None]
x_y_values *= 1e3
time_end = time.time()

print(x_y_values)
print(x_y_values.shape)
print(torch.sum(~torch.isnan(x_y_values)))
print(x_y_values[~torch.isnan(x_y_values)].max())
print(time_end - time_start)