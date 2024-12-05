import torch
import numpy as np

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

def expand_wav(inputs, dispersion_pixels, n_lambda = 28):
    """
        Input [bs, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
    """
    bs,row,col = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    
    output = torch.zeros(bs, n_lambda, row, col).float()#.cuda()
    for i in range(n_lambda):
        output[:, i, abs_shift[i, 0]: row - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: col - max_min[1] + abs_shift[i, 1]] = inputs[:, abs_shift[i, 0]: row - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: col - max_min[1] + abs_shift[i, 1]]
    return output

def shift_3d(inputs, dispersion_pixels):
    """
        Input [bs, n_wav, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
        Rolls the input along the row axis
    """
    [bs, nC, row, col] = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices    
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=int(abs_shift[i, 0]), dims=1) # Roll along row axis too
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=int(abs_shift[i, 1]), dims=2)
    return inputs


def shift_back_3d(inputs, dispersion_pixels):
    """
        Input [bs, n_wav, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
        Rolls input in the opposite direction
    """
    [bs, nC, row, col] = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices    
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*int(abs_shift[i, 0]), dims=1) # Roll along row axis too
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*int(abs_shift[i, 1]), dims=2)
    return inputs

def shift_back_mst(inputs, dispersion_pixels): 
    """
        Input [bs, n_wav, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H, W]
    """
    bs, nC, H, W = inputs.shape
    down_sample = nC//28
    rounded_disp = dispersion_pixels.round().int()//down_sample # Pixels to indices
    rounded_disp = torch.nn.functional.interpolate(rounded_disp.unsqueeze(0).unsqueeze(0).float(),
                                                   size=(rounded_disp.shape[0]*down_sample, 2), mode='bilinear').squeeze().round().int()
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    output = torch.zeros(bs, nC, H - max_min[0], W - max_min[1])
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(nC):
        output[:, i, :, :] = inputs[:, i, abs_shift[i, 0]: H - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: W - max_min[1] + abs_shift[i, 1]]
    return output


if __name__ == '__main__':
    image = torch.load("test_amici_n100_ov10.pt").flip(0,1) # H, W, nC
    print("Cube shape: ", image.shape)
    dispersion_pixels = torch.tensor([[  0.0000, -42.8502],
        [  0.0000, -42.2557],
        [  0.0000, -41.6673],
        [  0.0000, -41.0816],
        [  0.0000, -40.5001],
        [  0.0000, -39.9225],
        [  0.0000, -39.3504],
        [  0.0000, -38.7811],
        [  0.0000, -38.2166],
        [  0.0000, -37.6550],
        [  0.0000, -37.0984],
        [  0.0000, -36.5451],
        [  0.0000, -35.9960],
        [  0.0000, -35.4502],
        [  0.0000, -34.9087],
        [  0.0000, -34.3713],
        [  0.0000, -33.8370],
        [  0.0000, -33.3061],
        [  0.0000, -32.7790],
        [  0.0000, -32.2559],
        [  0.0000, -31.7363],
        [  0.0000, -31.2200],
        [  0.0000, -30.7073],
        [  0.0000, -30.1988],
        [  0.0000, -29.6919],
        [  0.0000, -29.1894],
        [  0.0000, -28.6916],
        [  0.0000, -28.1944],
        [  0.0000, -27.7028],
        [  0.0000, -27.2138],
        [  0.0000, -26.7279],
        [  0.0000, -26.2438],
        [  0.0000, -25.7652],
        [  0.0000, -25.2886],
        [  0.0000, -24.8154],
        [  0.0000, -24.3451],
        [  0.0000, -23.8779],
        [  0.0000, -23.4132],
        [  0.0000, -22.9531],
        [  0.0000, -22.4941],
        [  0.0000, -22.0379],
        [  0.0000, -21.5867],
        [  0.0000, -21.1368],
        [  0.0000, -20.6903],
        [  0.0000, -20.2465],
        [  0.0000, -19.8057],
        [  0.0000, -19.3680],
        [  0.0000, -18.9327],
        [  0.0000, -18.4992],
        [  0.0000, -18.0693],
        [  0.0000, -17.6420],
        [  0.0000, -17.2178],
        [  0.0000, -16.7959],
        [  0.0000, -16.3767],
        [  0.0000, -15.9590],
        [  0.0000, -15.5467],
        [  0.0000, -15.1344],
        [  0.0000, -14.7248],
        [  0.0000, -14.3186],
        [  0.0000, -13.9146],
        [  0.0000, -13.5120],
        [  0.0000, -13.1135],
        [  0.0000, -12.7162],
        [  0.0000, -12.3228],
        [  0.0000, -11.9302],
        [  0.0000, -11.5404],
        [  0.0000, -11.1539],
        [  0.0000, -10.7686],
        [  0.0000, -10.3854],
        [  0.0000, -10.0052],
        [  0.0000,  -9.6267],
        [  0.0000,  -9.2513],
        [  0.0000,  -8.8770],
        [  0.0000,  -8.5049],
        [  0.0000,  -8.1363],
        [  0.0000,  -7.7694],
        [  0.0000,  -7.4050],
        [  0.0000,  -7.0413],
        [  0.0000,  -6.6808],
        [  0.0000,  -6.3224],
        [  0.0000,  -5.9659],
        [  0.0000,  -5.6111],
        [  0.0000,  -5.2585],
        [  0.0000,  -4.9084],
        [  0.0000,  -4.5598],
        [  0.0000,  -4.2131],
        [  0.0000,  -3.8686],
        [  0.0000,  -3.5266],
        [  0.0000,  -3.1859],
        [  0.0000,  -2.8469],
        [  0.0000,  -2.5111],
        [  0.0000,  -2.1759],
        [  0.0000,  -1.8427],
        [  0.0000,  -1.5110],
        [  0.0000,  -1.1823],
        [  0.0000,  -0.8555],
        [  0.0000,  -0.5292],
        [  0.0000,  -0.2067],
        [  0.0000,   0.1161],
        [  0.0000,   0.4361],
        [  0.0000,   0.7555],
        [  0.0000,   1.0705],
        [  0.0000,   1.3863],
        [  0.0000,   1.6990],
        [  0.0000,   2.0104],
        [  0.0000,   2.3205],
        [  0.0000,   2.6284],
        [  0.0000,   2.9350],
        [  0.0000,   3.2392],
        [  0.0000,   3.5428],
        [  0.0000,   3.8435],
        [  0.0000,   4.1428],
        [  0.0000,   4.4421],
        [  0.0000,   4.7386],
        [  0.0000,   5.0335],
        [  0.0000,   5.3258],
        [  0.0000,   5.6193],
        [  0.0000,   5.9093],
        [  0.0000,   6.1972],
        [  0.0000,   6.4840],
        [  0.0000,   6.7710],
        [  0.0000,   7.0544],
        [  0.0000,   7.3372],
        [  0.0000,   7.6186],
        [  0.0000,   7.8981],
        [  0.0000,   8.1764],
        [  0.0000,   8.4531],
        [  0.0000,   8.7292],
        [  0.0000,   9.0037],
        [  0.0000,   9.2746],
        [  0.0000,   9.5456],
        [  0.0000,   9.8161],
        [  0.0000,  10.0847],
        [  0.0000,  10.3516],
        [  0.0000,  10.6173],
        [  0.0000,  10.8816],
        [  0.0000,  11.1437],
        [  0.0000,  11.4057],
        [  0.0000,  11.6664],
        [  0.0000,  11.9256],
        [  0.0000,  12.1826],
        [  0.0000,  12.4382],
        [  0.0000,  12.6929],
        [  0.0000,  12.9469],
        [  0.0000,  13.1995],
        [  0.0000,  13.4503],
        [  0.0000,  13.7007],
        [  0.0000,  13.9489],
        [  0.0000,  14.1966],
        [  0.0000,  14.4425],
        [  0.0000,  14.6875],
        [  0.0000,  14.9317],
        [  0.0000,  15.1749],
        [  0.0000,  15.4149],
        [  0.0000,  15.6556],
        [  0.0000,  15.8935],
        [  0.0000,  16.1329],
        [  0.0000,  16.3698],
        [  0.0000,  16.6044],
        [  0.0000,  16.8386],
        [  0.0000,  17.0725],
        [  0.0000,  17.3039],
        [  0.0000,  17.5349],
        [  0.0000,  17.7656],
        [  0.0000,  17.9935],
        [  0.0000,  18.2210],
        [  0.0000,  18.4468],
        [  0.0000,  18.6720],
        [  0.0000,  18.8972],
        [  0.0000,  19.1205],
        [  0.0000,  19.3427],
        [  0.0000,  19.5639],
        [  0.0000,  19.7843],
        [  0.0000,  20.0031],
        [  0.0000,  20.2207],
        [  0.0000,  20.4382],
        [  0.0000,  20.6551],
        [  0.0000,  20.8694],
        [  0.0000,  21.0829],
        [  0.0000,  21.2956],
        [  0.0000,  21.5077],
        [  0.0000,  21.7190],
        [  0.0000,  21.9295],
        [  0.0000,  22.1377],
        [  0.0000,  22.3463],
        [  0.0000,  22.5522],
        [  0.0000,  22.7598],
        [  0.0000,  22.9637],
        [  0.0000,  23.1682],
        [  0.0000,  23.3710],
        [  0.0000,  23.5732],
        [  0.0000,  23.7751],
        [  0.0000,  23.9756],
        [  0.0000,  24.1753],
        [  0.0000,  24.3737],
        [  0.0000,  24.5717],
        [  0.0000,  24.7686],
        [  0.0000,  24.9637],
        [  0.0000,  25.1594],
        [  0.0000,  25.3539],
        [  0.0000,  25.5468],
        [  0.0000,  25.7392],
        [  0.0000,  25.9314],
        [  0.0000,  26.1208],
        [  0.0000,  26.3116],
        [  0.0000,  26.4996],
        [  0.0000,  26.6878],
        [  0.0000,  26.8756],
        [  0.0000,  27.0628],
        [  0.0000,  27.2485],
        [  0.0000,  27.4325],
        [  0.0000,  27.6161],
        [  0.0000,  27.8000],
        [  0.0000,  27.9810],
        [  0.0000,  28.1646],
        [  0.0000,  28.3442],
        [  0.0000,  28.5237],
        [  0.0000,  28.7039],
        [  0.0000,  28.8827],
        [  0.0000,  29.0595],
        [  0.0000,  29.2366],
        [  0.0000,  29.4134],
        [  0.0000,  29.5882],
        [  0.0000,  29.7625],
        [  0.0000,  29.9369],
        [  0.0000,  30.1092],
        [  0.0000,  30.2809],
        [  0.0000,  30.4539],
        [  0.0000,  30.6243],
        [  0.0000,  30.7944],
        [  0.0000,  30.9636],
        [  0.0000,  31.1321],
        [  0.0000,  31.3002],
        [  0.0000,  31.4674],
        [  0.0000,  31.6340],
        [  0.0000,  31.7992],
        [  0.0000,  31.9646],
        [  0.0000,  32.1285],
        [  0.0000,  32.2929],
        [  0.0000,  32.4557],
        [  0.0000,  32.6178],
        [  0.0000,  32.7790],
        [  0.0000,  32.9408],
        [  0.0000,  33.1006],
        [  0.0000,  33.2608],
        [  0.0000,  33.4199],
        [  0.0000,  33.5775],
        [  0.0000,  33.7357],
        [  0.0000,  33.8932],
        [  0.0000,  34.0486],
        [  0.0000,  34.2042],
        [  0.0000,  34.3585],
        [  0.0000,  34.5134],
        [  0.0000,  34.6674],
        [  0.0000,  34.8205],
        [  0.0000,  34.9722],
        [  0.0000,  35.1248],
        [  0.0000,  35.2764],
        [  0.0000,  35.4267],
        [  0.0000,  35.5765],
        [  0.0000,  35.7255],
        [  0.0000,  35.8751],
        [  0.0000,  36.0231],
        [  0.0000,  36.1704],
        [  0.0000,  36.3175],
        [  0.0000,  36.4633],
        [  0.0000,  36.6095],
        [  0.0000,  36.7540],
        [  0.0000,  36.8991],
        [  0.0000,  37.0434],
        [  0.0000,  37.1864],
        [  0.0000,  37.3295],
        [  0.0000,  37.4720],
        [  0.0000,  37.6143],
        [  0.0000,  37.7542],
        [  0.0000,  37.8948],
        [  0.0000,  38.0352],
        [  0.0000,  38.1745],
        [  0.0000,  38.3133],
        [  0.0000,  38.4517]])
    
    bigger_cube = torch.zeros((256, 512, 280))
    bigger_cube[:, 128:384, :] = image
    dispersion_pixels = dispersion_pixels[::10,:]
    print("Dispersion : ", dispersion_pixels.max() - dispersion_pixels.min())
    middle_pos = [128, 256]
    texture_size = [256, 256]
    acq = extract_acq_from_cube(bigger_cube, dispersion_pixels, middle_pos, texture_size)
    print("Acq shape: ", acq.shape)

    shift_back_acq = shift_back(acq.unsqueeze(0), dispersion_pixels, n_lambda = 28)

    print(shift_back_acq.shape)

    shift_shift_back_acq = shift(shift_back_acq, dispersion_pixels)

    print(shift_shift_back_acq.shape)

    expand_wav_acq = expand_wav(acq.unsqueeze(0), dispersion_pixels, n_lambda = 28)

    print(expand_wav_acq.shape)

    shift_3d_acq = shift_3d(expand_wav_acq, dispersion_pixels)

    print(shift_3d_acq.shape)
    
    shift_back_3d_acq = shift_back_3d(shift_3d_acq, dispersion_pixels)

    print(shift_back_3d_acq.shape)

    print("Shift back shift == original? ", torch.sum(expand_wav_acq == shift_back_3d_acq)/torch.numel(expand_wav_acq))

    shift_back_mst_acq = shift_back_mst(expand_wav_acq, dispersion_pixels)

    print(shift_back_mst_acq.shape)

    mask = torch.rand(512, 512) < 0.5
    torch.save(mask, "mask.pt")

    #import torch
    import torch.nn.functional as F

    def airy_disk(wavelength, na, pixel_size, grid_size, magnification = 1):
        """
        Compute the Airy disk pattern.

        Parameters:
            wavelength (float): Wavelength of the light.
            na (float): Angle of the numerical aperture (in radians).
            pixel_size (float): Size of the pixel.
            grid_size (int): Size of the grid for the computation.
            magnification (float): Magnification factor of the Airy disk.
        Returns:
            torch.Tensor: 2D tensor representing the Airy disk pattern.
        """
        # Create a grid of coordinates
        x = torch.linspace(-grid_size // 2 +1, grid_size// 2 , grid_size) * pixel_size
        y = torch.linspace(-grid_size // 2 +1, grid_size// 2 , grid_size) * pixel_size
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Calculate the radial distance from the center
        R = torch.sqrt(X**2 + Y**2)

        # Compute the Airy disk pattern
        k = 1/magnification * torch.pi* 2*R*torch.tan(torch.as_tensor(na))/ wavelength
        airy_pattern = (2*torch.special.bessel_j1(k) / k).pow(2)
        airy_pattern[R == 0] = 1  # Handle the singularity at the center

        # Normalize the pattern
        airy_pattern /= airy_pattern.sum()

        return airy_pattern

    # Example usage
    wavelength = 520e-6
    na = 0.05
    grid_size = 7
    pixel_size = 10e-3

    airy_disk_pattern = airy_disk(wavelength, na, pixel_size, grid_size, magnification = 1.5)
    import matplotlib.pyplot as plt
    plt.imshow(airy_disk_pattern, cmap='viridis')
    plt.colorbar()
    plt.figure()
    plt.plot(torch.linspace(-grid_size // 2 +1, grid_size// 2 , grid_size) * pixel_size, airy_disk_pattern[grid_size//2, :], 'r')
    plt.show()

    wavelengths = torch.linspace(450, 650, 280)
    airy_disk_kernel = torch.zeros(wavelengths.shape[0], 1, grid_size, grid_size)
    for i in range(wavelengths.shape[0]):
        airy_disk_kernel[i, 0, :, :] = airy_disk(wavelengths[i]*1e-6, na, pixel_size, grid_size, magnification = 2)
    
    convolved = F.conv2d(image.permute(2, 0, 1).unsqueeze(0), airy_disk_kernel, padding = grid_size//2, groups=28*10)
    print(convolved.shape)

    convolved_1 = F.conv2d(image.permute(2, 0, 1)[25, ...].unsqueeze(0).unsqueeze(0), airy_disk_kernel[25, 0, ...].unsqueeze(0).unsqueeze(0), padding = grid_size//2)
    print(torch.sum(convolved[0, 25, ...] == convolved_1[0,0,...])/(256*256))

    plt.figure()
    plt.imshow(image.sum(-1), cmap='viridis')
    plt.colorbar()
    plt.figure()
    plt.imshow(convolved[0, :, :, :].sum(0), cmap='viridis')
    plt.colorbar()
    plt.show()

    texture = torch.arange(4*4*4*4).reshape(4, 4, 4, 4).float()
    position = torch.tensor([[1, 3, 0, 1],
                             [3, 2, 2, 3],
                             [0, 2, 0 , 0],
                             [0, 0, 0, 0]]).int()
    print(texture[position[:,0], position[:,1]])
    
    pos_t = position.t()
    print(texture[pos_t[0], pos_t[1], pos_t[2], pos_t[3]])

    texture = torch.zeros((3, 4, 5, 2))
    print(texture + torch.tensor([1, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0))
