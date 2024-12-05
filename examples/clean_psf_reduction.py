import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import glob
from nmf import run_nmf

limit = 2.7
nb_pts = 50

dir_name = "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/psfs/"

list_of_files = list(glob.glob(dir_name + f"psf_field_*_lim_{limit}_pts_{nb_pts}.npy"))
list_of_files.sort()
n_lam = len(list_of_files)

def compute_A(list_of_files, nb_pts, n_lam, pixel_size=0.005, kernel_size=11):
    """
    Compute the matrix A for PSF reduction. It is made from the PSFs in files list_of_files.

    Args:
        list_of_files (list): List of file paths containing PSF matrices.
        nb_pts (int): Number of points in each dimension of the PSF matrices.
        n_lam (int): Number of wavelengths.
        pixel_size (float, optional): Pixel size. Defaults to 0.005.
        kernel_size (int, optional): Size of the kernel. Defaults to 11.

    Returns:
        torch.Tensor: Matrix A for PSF reduction.
    """

    A = torch.zeros((nb_pts * nb_pts * n_lam, kernel_size * kernel_size))
    for f, file_path in enumerate(list_of_files):
        PSF_mat = np.load(file_path)  # (nb_pts, nb_pts, <=nb_rays, 2)

        # Flip the PSF matrix and transpose it so that the x axis is the first axis and the y axis is the second axis
        # The x axis is directed to the right and the y axis is oriented upwards
        PSF_mat = np.flip(PSF_mat, (0,))
        PSF_mat = np.transpose(PSF_mat, (1, 0, 2, 3))
        for i in range(PSF_mat.shape[0]):
            for j in range(PSF_mat.shape[1]):
                ps = PSF_mat[i, j, ...]
                ps = np.nan_to_num(ps, nan=0)
                ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1, 2)
                bins_i, bins_j, _ = find_bins(ps, pixel_size, kernel_size)
                hist = torch.histogramdd(
                    torch.from_numpy(ps).float().flip(1),
                    bins=(bins_j, bins_i),
                    density=False,
                ).hist
                A[f * nb_pts * nb_pts + i * PSF_mat.shape[1] + j, :] = hist.flatten()
    A = torch.nan_to_num(A, nan=0.0)
    torch.save(
        A,
        f"/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/A_{limit}lim_{nb_pts}pts.pt",
    )
    return A

def compute_centroid(points):
    """
    Compute the centroid of a set of points.

    Args:
        points (torch.Tensor or numpy.ndarray): The input points. If a numpy array is provided, it will be converted to a torch tensor.

    Returns:
        torch.Tensor: The centroid of the input points.

    """
    points = points if type(points) is torch.Tensor else torch.from_numpy(points)
    return torch.mean(points, dim=0)

def find_bins(points, size_bin, nb_bins=11):
    """
    Find the bins for a given set of points.

    Args:
        points (torch.Tensor): The input points.
        size_bin (float): The size of each bin.
        nb_bins (int, optional): The number of bins. Defaults to 11.

    Returns:
        tuple: A tuple containing the x-coordinates of the bins, the y-coordinates of the bins, and the centroid of the points.
    """
    bins_i = torch.zeros((nb_bins+1))
    bins_j = torch.zeros((nb_bins+1))
    centroid = compute_centroid(points)

    for i in range(nb_bins+1):
        bins_i[i] = centroid[0] - size_bin/2 - (nb_bins//2)*size_bin + i*size_bin
        bins_j[i] = centroid[1] - size_bin/2 - (nb_bins//2)*size_bin + i*size_bin

    return bins_i, bins_j, centroid

def generate_psf_at_pos(pos_x, pos_y, pos_lambda, limit, nb_pts, sub_G, sub_W, kernel_size=11, show=True):
    """
    Generate the Point Spread Function (PSF) at a given position.

    Args:
        pos_x (float): The x-coordinate of the position.
        pos_y (float): The y-coordinate of the position.
        pos_lambda (float): The wavelength of the position.
        limit (float): The limit of the pattern.
        nb_pts (int): The number of points in the pattern.
        sub_G (torch.Tensor): The sub_G tensor.
        sub_W (torch.Tensor): The sub_W tensor.
        kernel_size (int, optional): The size of the PSF kernel. Defaults to 11.
        show (bool, optional): Whether to display the PSF image. Defaults to True.

    Returns:
        torch.Tensor: The PSF at the given position.
    """

    pos_x = torch.tensor([pos_x])
    pos_y = torch.tensor([pos_y])
    pos_lambda = torch.tensor([pos_lambda])

    # Calculate the indices for bilinear interpolation
    id_x = (pos_x+limit)/(2*limit)*(nb_pts-1)
    id_y = (nb_pts-1) - (pos_y+limit)/(2*limit)*(nb_pts-1)  # Invert y axis because top left is (0,0) in the PSF matrix and corresponds to +limit in the psf position (-lim -> nbpts ; +lim -> 0)

    id_x_close = torch.round((torch.tensor([pos_x])+limit)/(2*limit)*(nb_pts-1), decimals=3)  # Trick to avoid rounding errors
    id_y_close = torch.round((nb_pts-1) - (torch.tensor([pos_y])+limit)/(2*limit)*(nb_pts-1), decimals=3)  # Trick to avoid rounding errors

    # Get the corners for bilinear interpolation
    corner_1 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.floor(id_y_close)*nb_pts + torch.floor(id_x_close)).int(), ...]  # NW
    corner_2 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.ceil(id_y_close)*nb_pts + torch.floor(id_x_close)).int(), ...]  # SW
    corner_3 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.floor(id_y_close)*nb_pts + torch.ceil(id_x_close)).int(), ...]  # NE
    corner_4 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.ceil(id_y_close)*nb_pts + torch.ceil(id_x_close)).int(), ...]  # SE

    # Perform bilinear interpolation to compute the PSF at id_x, id_y
    delta_fx = id_x - torch.floor(id_x_close)
    delta_fy = id_y - torch.floor(id_y_close)
    delta_cx = 1 - delta_fx
    delta_cy = 1 - delta_fy

    weight_1 = delta_cx*delta_fy
    weight_2 = delta_cx*delta_cy
    weight_3 = delta_fx*delta_fy
    weight_4 = delta_fx*delta_cy

    w_PSF_at_pos = corner_1*weight_1 + corner_2*weight_2 + corner_3*weight_3 + corner_4*weight_4

    PSF_at_pos = w_PSF_at_pos @ sub_W

    if show:
        plt.figure()
        plt.imshow(PSF_at_pos.reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest')
        plt.show()

    return PSF_at_pos

def compute_graph_heatmap_from_database(list_of_files, limit, nb_pts, n_lam = 55, sub_folder = "database_for_comparison", K=None, k_min=20, k_max=122, nb_compar=2000,
                                        reduction_factor=1, pixel_size=0.02, kernel_size=11, method = 'svd'):
    """
    Computes the graph heatmap from a database of PSF files.

    Args:
    - list_of_files: List of PSF file paths.
    - limit: Limit of the PSF field.
    - nb_pts: Number of points in the PSF field.
    - n_lam: Number of wavelengths.
    - sub_folder: Subfolder name for the database.
    - K: List of compactness values.
    - k_min: Minimum compactness value.
    - k_max: Maximum compactness value.
    - nb_compar: Number of comparisons.
    - reduction_factor: Reduction factor for the number of points.
    - pixel_size: Pixel size.
    - kernel_size: Kernel size.
    - method: Method for computing the graph heatmap.

    Returns:
    - list_rmse: Array of RMSE values.
    """

    try:
        A = torch.load(f"/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/A_{limit}lim_{nb_pts}pts.pt")
    except:
        A = compute_A(list_of_files, nb_pts, n_lam, pixel_size= pixel_size, kernel_size=kernel_size)
    
    # Used to subsample the PSF field, if we want to reduce the number of points
    if reduction_factor != 1:
        A = A.reshape(n_lam, nb_pts, nb_pts, kernel_size*kernel_size)
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        sub_sampling = f(int(nb_pts//reduction_factor), nb_pts)
        A = A.index_select(1, torch.tensor(sub_sampling)).index_select(2, torch.tensor(sub_sampling))
        A = A.flatten(start_dim=0, end_dim=2)
        nb_pts = int(nb_pts//reduction_factor)
    
    if method == 'svd':
        U, S, V = torch.linalg.svd(A, full_matrices=False)
    elif method == 'svd_squared':
        U, S, V = torch.linalg.svd(torch.sqrt(A), full_matrices=False)
    
    list_of_psf = list(glob.glob(dir_name + f"{sub_folder}/*.npy"))
    
    # Initialize the number of comparisons and the range of compactness values
    if nb_compar is None:
        nb_compar = len(list_of_psf)
    if K is None:
        K = list(range(k_min, k_max))
    else:
        K = list(K)
    
    # Initialize the storage objects
    list_rmse = np.zeros((len(K),min(nb_compar, len(list_of_psf))))

    heatmap_setup_indep = np.zeros((len(K), nb_pts, nb_pts, n_lam, (2*nb_compar)//(nb_pts*nb_pts))) - 1
    heatmap_setup_same = np.zeros((len(K), nb_pts, nb_pts, n_lam, (2*nb_compar)//(nb_pts*nb_pts))) - 1
    heatmap_setup_false = np.zeros((len(K), nb_pts, nb_pts, n_lam, (2*nb_compar)//(nb_pts*nb_pts))) - 1

    time_start = time.time()

    # Loop over the compactness values
    for k, compactness in enumerate(K):
        if 'svd' in method:
            sub_U = U[:, :compactness]
            sub_S = S[:compactness]
            sub_V = V[:compactness, :]

            sub_G = sub_U @ torch.diag(sub_S)
            sub_W = sub_V

        # Run the NMF algorithm for the corresponding K value
        elif method == 'nmf':
            sub_G, sub_W, err = run_nmf(A, n_components=compactness)
            sub_G = torch.from_numpy(sub_G)
            sub_W = torch.from_numpy(sub_W)

        # Plotting the PSFs generating family
        if compactness < 6:
            fig, ax = plt.subplots(1, compactness)
        else:
            fig, ax = plt.subplots(compactness//6,6)
        plot_min = torch.min(sub_W).numpy()
        plot_max = torch.max(sub_W).numpy()
        cmap = 'seismic' if 'svd' in method else 'viridis'
        vmin = plot_min
        vmax = plot_max
        vmin = min(vmin, -vmax)
        vmax = -vmin
        for i in range(compactness):
            if compactness < 6:
                bar = ax[i].imshow(sub_W[i,:].reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
                ax[i].get_xaxis().set_visible(False)
                ax[i].get_yaxis().set_visible(False)
            else:
                bar = ax[i // 6, i%6].imshow(sub_W[i,:].reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
                ax[i // 6, i%6].get_xaxis().set_visible(False)
                ax[i // 6, i%6].get_yaxis().set_visible(False)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(bar, cax=cbar_ax)
        plt.show()

        max_neg_proportion = 0
        worst_neg_PSF = torch.zeros((kernel_size, kernel_size))
        worst_neg_hist = torch.zeros((kernel_size, kernel_size))

        # Compute the RMSE comparisons for each PSF in the database
        for i, file_path in enumerate(list_of_psf[:nb_compar]):
            # Parse the file path to get the position and wavelength
            PSF_mat = np.load(file_path)
            parsing = file_path.split('/')[-1].split("_")
            x_pos = float(parsing[1])
            y_pos = float(parsing[3])
            lambda_pos = int(parsing[5][:-4])

            # Extract the PSF matrix and remove the zero values
            ps = PSF_mat
            ps = np.nan_to_num(ps, nan=0.0)
            ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
            
            # Compute the histogram of the PSF matrix
            bins_i, bins_j, centroid = find_bins(ps, pixel_size, kernel_size)
            hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=(bins_j, bins_i), density=False).hist
            hist = hist.T.numpy()

            # Generate the PSF at the given position
            PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, sub_G, sub_W, kernel_size=kernel_size, show=False)
            
            # In that method, the reconstructed PSF corresponds to the square root of the true PSF
            if method=='svd_squared':
                PSF_at_pos = torch.square(PSF_at_pos)
            
            # Keep track of the proportion of negative values in the PSF with the SVD method
            neg_sum = - torch.sum(PSF_at_pos[PSF_at_pos < 0])
            abs_sum = torch.sum(torch.abs(PSF_at_pos))
            PSF_at_pos_with_neg = PSF_at_pos

            # Clamp the PSF values to be positive
            PSF_at_pos = torch.clamp(PSF_at_pos, min=0.0)
            PSF_at_pos /= torch.sum(PSF_at_pos) # Normalize the PSF
            PSF_at_pos = PSF_at_pos.reshape(kernel_size,kernel_size).T.numpy()

            # Compute the RMSE between the histogram and the PSF
            mask = np.logical_or(hist > 0, PSF_at_pos > 0)
            rmse_mask = np.sqrt(np.mean(((hist - PSF_at_pos)[mask])**2)) # RMSE only on the non-zero values

            rmse = np.sqrt(np.mean(((hist - PSF_at_pos))**2))
            list_rmse[k, i] = rmse
            
            # Keep track of the PSF with the worst negative values
            if 'svd' in method and neg_sum/abs_sum > max_neg_proportion:
                max_neg_proportion = neg_sum/abs_sum
                worst_neg_PSF = PSF_at_pos_with_neg
                worst_neg_hist = hist

            # Compute the RMSE for the normalized PSFs
            hist_same = hist/ps.shape[0]
            PSF_at_pos_same = PSF_at_pos/ps.shape[0]

            PSF_at_pos_indep = PSF_at_pos/np.sum(PSF_at_pos)

            rmse_same = np.sqrt(np.mean((hist_same - PSF_at_pos_same)**2))
            rmse_indep = np.sqrt(np.mean((hist_same - PSF_at_pos_indep)**2))
            
            # Compute the indices for the heatmap
            id_x = torch.round((torch.tensor([x_pos])+limit)/(2*limit)*(nb_pts-1), decimals=1).int()
            id_y = torch.round((nb_pts-1) - (torch.tensor([y_pos])+limit)/(2*limit)*(nb_pts-1), decimals=1).int()

            # Store the RMSE values in the heatmap
            ind_same = np.where(heatmap_setup_same[k, id_y, id_x, lambda_pos, :]==-1)[0][0]
            ind_indep = np.where(heatmap_setup_indep[k, id_y, id_x, lambda_pos, :]==-1)[0][0]
            ind_false = np.where(heatmap_setup_false[k, id_y, id_x, lambda_pos, :]==-1)[0][0]
            
            heatmap_setup_same[k, id_y, id_x, lambda_pos, ind_same] = rmse_same
            heatmap_setup_indep[k, id_y, id_x, lambda_pos, ind_indep] = rmse_indep
            heatmap_setup_false[k, id_y, id_x, lambda_pos, ind_false] = rmse

        print(f"Finished for K={compactness} in {time.time() - time_start:.3f} seconds")
        print(f"Mean unnormalized RMSE: {np.mean(list_rmse[k, :]):.3f}")

        # Plot the PSF with the worst negative values
        if 'svd' in method:
            fig, ax = plt.subplots(1,2)
            plot_min = torch.min(worst_neg_PSF).numpy()
            plot_max = torch.max(worst_neg_PSF).numpy()
            cmap = 'seismic' if 'svd' in method else 'viridis'
            vmin = plot_min
            vmax = plot_max
            vmin = min(vmin, -vmax)
            vmax = -vmin
            ax[0].imshow(worst_neg_hist.reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
            bar = ax[1].imshow(worst_neg_PSF.reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(bar, cax=cbar_ax)
            fig.suptitle(f"Max proportion of negative values: {max_neg_proportion*100:.2f}%")
            plt.show()


    if 'svd' in method:
        prefix = "rmse"
    elif method == 'nmf':
        prefix = "rmse_nmf"
    
    if "regular" in sub_folder:
        prefix = "regular_" + prefix

    rmse_name = f"{prefix}_comparison_{n_lam}w_{limit}lim_{nb_pts}pts.npy"
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + rmse_name, list_rmse)

    heatmap_false_name = f"{prefix}_heatmap_{n_lam}w_{limit}lim_{nb_pts}pts.npy"
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + heatmap_false_name, heatmap_setup_false)

    heatmap_same_name = f"{prefix}_heatmap_normalized_same_{n_lam}w_{limit}lim_{nb_pts}pts.npy"
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + heatmap_same_name, heatmap_setup_same)

    heatmap_indep_name = f"{prefix}_heatmap_normalized_indep_{n_lam}w_{limit}lim_{nb_pts}pts.npy"
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + heatmap_indep_name, heatmap_setup_indep)

    heatmap_setup_false[heatmap_setup_false == -1] = np.nan
    heatmap_setup_false = np.nanmean(heatmap_setup_false, axis=(-2, -1))

    plt.figure()
    plt.imshow(heatmap_setup_false[0, ...], origin='lower', interpolation='nearest', extent=[-limit, limit, -limit, limit])
    plt.title(f"Mean RMSE for K={K}")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    clb = plt.colorbar()
    clb.ax.set_title(f"RMSE for {ps.shape[0]} rays")
    plt.show()
    return list_rmse

def plot_rmse(limit, nb_pts, w=55, file_path="/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/", data='comparison', method='svd', grid_type="regular", measure='mean'):
    """
    Plot the root mean square error (RMSE) values.

    Parameters:
    - limit (float): The limit of the heatmap extent.
    - nb_pts (int): The number of points in the heatmap.
    - w (int, optional): The value of w for the heatmap. Default is 55.
    - file_path (str, optional): The file path where the heatmap data is stored. Default is "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/".
    - data (str, optional): The type of data to plot. Can be 'comparison' or 'compression'. Default is 'comparison'.
    - method (str, optional): The method used to calculate the heatmap. Can be 'svd' or 'nmf'. Default is 'svd'.
    - grid_type (str, optional): The type of grid used for the heatmap. Can be 'regular' or 'irregular'. Default is 'regular'.
    - measure (str, optional): The measure used to calculate the heatmap. Can be 'mean', 'median', 'max', or 'min'. Default is 'mean'.

    Returns:
        None
    """
    if method == 'svd':
        prefix = "rmse"
    elif method == 'nmf':
        prefix = "rmse_nmf"

    if grid_type == "regular":
        prefix = "regular_" + prefix

    if data == 'comparison':
        file_name = f"{prefix}_comparison_{w}w_{limit}lim_{nb_pts}pts.npy"
    elif data == 'compression':
        file_name = f"{prefix}_compression_loss_{w}w_{limit}lim_{nb_pts}pts.npy"
    rmse_list = np.load(file_path + file_name)
    if measure == 'mean':
        rmse_list = np.mean(rmse_list, axis=1)
    elif measure == 'median':
        rmse_list = np.median(rmse_list, axis=1)
    elif measure == 'max':
        rmse_list = np.max(rmse_list, axis=1)
    elif measure == 'min':
        rmse_list = np.min(rmse_list, axis=1)
    plt.figure()
    plt.plot(list(range(3, rmse_list.shape[0]+3)), rmse_list)
    plt.title(f"RMSE w.r.t. K for {measure} value")
    plt.xlabel("Value of K")
    plt.ylabel("RMSE")
    plt.show()

def plot_heatmap(limit, nb_pts, K=60, w=55, file_path="/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/", measure='mean', method='svd', grid_type="regular", normalize='same'):
    """
    Plot a heatmap of the root mean square error (RMSE) values.

    Parameters:
    - limit (float): The limit of the heatmap extent.
    - nb_pts (int): The number of points in the heatmap.
    - K (int, optional): The value of K for the heatmap. Default is 60.
    - w (int, optional): The value of w for the heatmap. Default is 55.
    - file_path (str, optional): The file path where the heatmap data is stored. Default is "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/".
    - measure (str, optional): The measure used to calculate the heatmap. Can be 'mean', 'median', 'max', or 'min'. Default is 'mean'.
    - method (str, optional): The method used to calculate the heatmap. Can be 'svd' or 'nmf'. Default is 'svd'.
    - grid_type (str, optional): The type of grid used for the heatmap. Can be 'regular' or 'irregular'. Default is 'regular'.
    - normalize (str, optional): The normalization method used for the heatmap. Can be 'same', 'indep', or None. Default is 'same'.

    Returns:
        None
    """
    # Function code here
def plot_heatmap(limit, nb_pts, K=60, w=55, file_path = "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/", measure='mean', method = 'svd', grid_type = "regular", normalize='same'):
    if method == 'svd':
        prefix = "rmse"
    elif method == 'nmf':
        prefix = "rmse_nmf"

    if grid_type == "regular":
        prefix = "regular_" + prefix

    if normalize in ['same', 'indep']:
        file_name = f"{prefix}_heatmap_normalized_{normalize}_{w}w_{limit}lim_{nb_pts}pts.npy"
    else:
        file_name = f"{prefix}_heatmap_{w}w_{limit}lim_{nb_pts}pts.npy"
    heatmap = np.load(file_path + file_name)
    heatmap[heatmap == -1] = np.nan
    if measure == 'mean':
        heatmap = np.nanmean(heatmap, axis=(-2, -1))
    elif measure == 'median':
        heatmap = np.nanmedian(heatmap, axis=-1)
    elif measure == 'max':
        heatmap = np.nanmax(heatmap, axis=-1)
    elif measure == 'min':
        heatmap = np.nanmin(heatmap, axis=-1)
    
    plt.figure()
    plt.imshow(heatmap[K, ...], origin='lower', interpolation='nearest', extent=[-limit, limit, -limit, limit])
    plt.title(f"RMSE for K={K} and {measure} value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()

kernel_size = 11
pixel_size = 0.005

compute_graph_heatmap_from_database(list_of_files, limit, nb_pts, n_lam, sub_folder = "50_2.56_regular_database_for_comparison", K=list(range(12,13)), method = 'svd', nb_compar=None,
                                    pixel_size=pixel_size, kernel_size=kernel_size)
#plot_rmse(limit, nb_pts, w=n_lam, data='comparison', method = 'svd', grid_type = "regular", measure='mean')
#plot_heatmap(limit, nb_pts, K=0, w=n_lam, measure='mean', method='svd', grid_type = "regular", normalize = 'indep')
#plot_heatmap(limit, nb_pts, K=0, w=n_lam, measure='mean', method='svd', grid_type = "regular", normalize = 'same')
#plot_heatmap(limit, nb_pts, K=0, w=n_lam, measure='mean', method='svd', grid_type = "reregular", normalize = False)