import torch
from main_class import *

optics_model_file_path = "./system_amici.yml"
lens_group = HSSystem(config_file_path = optics_model_file_path)
        
nb_rays = 20
nb_w = 112
nb_pixels = 128

lens_group.system[0].film_size = [nb_pixels, nb_pixels]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wavelengths = torch.linspace(450, 650, nb_w).to(device)

z0 = torch.tensor([lens_group.system[-1].d_sensor*torch.cos(lens_group.system[-1].theta_y*np.pi/180).item() + lens_group.system[-1].origin[-1]]).cpu().detach().item()

big_uv_1 = torch.load('./rays_amici.pt')
big_mask_1 = torch.load('./rays_valid_amici.pt')
if nb_pixels < 512 or nb_w < 112 or nb_rays < 20:
    big_uv = big_uv_1[:nb_w, :nb_rays,:nb_pixels*nb_pixels,:].clone()
    big_mask = big_mask_1[:nb_w, :nb_rays,:nb_pixels*nb_pixels].clone()
    del big_uv_1, big_mask_1
else:
    big_uv = big_uv_1
    big_mask = big_mask_1

torch.cuda.empty_cache()

print(big_uv.shape)

texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
texture = 1000*torch.tensor(texture).repeat_interleave(4).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
mask = np.zeros((256, 256), dtype=np.float32)
mask[:, 128] = 1
texture = np.multiply(texture, mask[:,:,np.newaxis]) 

texture = torch.from_numpy(texture).float().to(device)
texture = texture[:nb_pixels, :nb_pixels, :nb_w].unsqueeze(0) # batch

print(f"Memory used post network: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}Mb")

time_start_all = time.time()
image = lens_group.render_all_based_on_saved_pos(big_uv = big_uv, big_mask = big_mask, texture = texture, nb_rays=nb_rays, wavelengths = wavelengths,
                            z0=z0)

time_end_all = time.time()
print("Time for rendering all: ", time_end_all - time_start_all)

time_start_batch = time.time()
image = lens_group.render_batch_based_on_saved_pos(big_uv = big_uv, big_mask = big_mask, texture = texture, nb_rays=nb_rays, wavelengths = wavelengths,
                            z0=z0)

time_end_batch = time.time()
print("Time for rendering batch: ", time_end_batch - time_start_batch)





