import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import glob

limit = 2.7
nb_pts = 20

dir_name = "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/psfs/"

list_of_files = list(glob.glob(dir_name + f"psf_field_*_lim_{limit}_pts_{nb_pts}.npy"))
list_of_files.sort()
n_lam = len(list_of_files)

def compute_A(list_of_files, nb_pts, n_lam, pixel_size = 0.02, kernel_size=11):
    A = torch.zeros((nb_pts*nb_pts*n_lam, kernel_size*kernel_size))
    for f, file_path in enumerate(list_of_files):
        PSF_mat = np.load(file_path) # (nb_pts, nb_pts, <=nb_rays, 2)

        PSF_mat = np.flip(PSF_mat, (0,))
        PSF_mat = np.transpose(PSF_mat, (1, 0, 2, 3))
        for i in range(PSF_mat.shape[0]):
            for j in range(PSF_mat.shape[1]):
                ps = PSF_mat[i, j, ...]
                ps = np.nan_to_num(ps, nan=0)
                ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                #hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=kernel_size, density=False).hist
                bins_i, bins_j, _ = find_bins(ps, pixel_size, kernel_size)
                hist = torch.histogramdd(torch.from_numpy(ps).float().flip(1), bins=(bins_j, bins_i), density=False).hist
                """ if (f==19) and (i ==4):
                    plt.scatter(ps[...,1], ps[...,0], s=0.1)
                    plt.title(f"Position: x={j}")
                    plt.savefig('/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/psfs_optim/' + f"image_{j}.png", format='png')
                    plt.show() """
                A[f*nb_pts*nb_pts + i*PSF_mat.shape[1] + j, :] = hist.flatten()
    #A = A[~torch.any(A.isnan(), dim=1), :]
    A = torch.nan_to_num(A, nan=0.0)
    return A

def compute_A_pos_detector(list_of_files, nb_pts, limit, n_lam, pixel_size = 0.02, kernel_size=11):
    first_w = torch.from_numpy(np.load(list_of_files[0]))
    min_x, max_x = torch.min(first_w[...,1]), torch.max(first_w[...,1])
    min_y, max_y = torch.min(first_w[...,0]), torch.max(first_w[...,0])
    last_w = torch.from_numpy(np.load(list_of_files[-1]))
    min_x = min(min_x, torch.min(last_w[...,1]))
    max_x = max(max_x, torch.max(last_w[...,1]))
    min_y = min(min_y, torch.min(last_w[...,0]))
    max_y = max(max_y, torch.max(last_w[...,0]))
    dist_between_points = 2*limit/(nb_pts-1)
    id_min_x = torch.round((min_x+limit)/dist_between_points, decimals=1).int()
    id_max_x = torch.round((max_x+limit)/dist_between_points, decimals=1).int()
    id_min_y = torch.round((min_y+limit)/dist_between_points, decimals=1).int()
    id_max_y = torch.round((max_y+limit)/dist_between_points, decimals=1).int()

    print(f"Min x: {id_min_x}; {min_x:.3f}mm, Max x: {id_max_x}; {max_x:.3f}mm")
    print(f"Min y: {id_min_y}; {min_y:.3f}mm, Max y: {id_max_y}; {max_y:.3f}mm")
    nb_pts_x = id_max_x - id_min_x + 1
    nb_pts_y = id_max_y - id_min_y + 1

    dispersion_direction = 'x' if nb_pts_x > nb_pts_y else 'y'
    A = torch.zeros((nb_pts_x*nb_pts_y*n_lam, kernel_size*kernel_size))
    for f, file_path in enumerate(list_of_files):
        PSF_mat = np.load(file_path) # (nb_pts, nb_pts, <=nb_rays, 2)
        list_forgotten = [(x,y) for x in range(nb_pts_x) for y in range(nb_pts_y)]
        PSF_mat = np.flip(PSF_mat, (0,))
        PSF_mat = np.transpose(PSF_mat, (1, 0, 2, 3))
        for i in range(PSF_mat.shape[0]): # y
            for j in range(PSF_mat.shape[1]): # x
                ps = PSF_mat[i, j, ...]
                ps = np.nan_to_num(ps, nan=0)
                ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                #hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=kernel_size, density=False).hist
                bins_i, bins_j, centroid = find_bins(ps, pixel_size, kernel_size)
                hist = torch.histogramdd(torch.from_numpy(ps).float().flip(1), bins=(bins_j, bins_i), density=False).hist
                
                id_x = torch.round((centroid[1]+limit)/dist_between_points + abs(id_min_x)).int()
                #if j == 0:
                #    id_y = torch.round((nb_pts_y-1) - ((centroid[0]+limit)/dist_between_points + abs(id_min_y))).int()
                id_y = torch.round((nb_pts_y-1) - ((centroid[0]+limit)/dist_between_points + abs(id_min_y))).int()
                list_forgotten.remove((id_x, id_y))
                if f == 17 and id_y==21:
                    print(id_x)
                    print(f"id non rounded: {(nb_pts_y-1) - ((centroid[0]+limit)/dist_between_points + abs(id_min_y))}")
                    print(centroid[0])
                #id_y = torch.round((centroid[0]+limit)/dist_between_points + abs(id_min_y)).int()
                #print(id_y)
                A[f*nb_pts_x*nb_pts_y + id_y*nb_pts_x + id_x, :] = hist.flatten()
        for elem in list_forgotten:
            forgot_x, forgot_y = elem
            if dispersion_direction == 'x':
                if forgot_x-1 >=0:
                    if forgot_x+1 < nb_pts_x:
                        A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x, :] = 1/2 * (A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x-1, :] + A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x+1, :])
                    else:
                        A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x, :] = A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x-1, :]
                else:
                    A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x, :] = A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x+1, :]
            else:
                if forgot_y-1 >= 0:
                    if forgot_y+1 < nb_pts_y:
                        A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x, :] = 1/2 * (A[f*nb_pts_x*nb_pts_y + (forgot_y-1)*nb_pts_x + forgot_x, :] + A[f*nb_pts_x*nb_pts_y + (forgot_y+1)*nb_pts_x + forgot_x, :])
                    else:
                        A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x, :] = A[f*nb_pts_x*nb_pts_y + (forgot_y-1)*nb_pts_x + forgot_x, :]
                else:
                    A[f*nb_pts_x*nb_pts_y + forgot_y*nb_pts_x + forgot_x, :] = A[f*nb_pts_x*nb_pts_y + (forgot_y+1)*nb_pts_x + forgot_x, :]
    #A = A[~torch.any(A.isnan(), dim=1), :]
    A = torch.nan_to_num(A, nan=0.0)
    
    return A, nb_pts_x, nb_pts_y, id_min_x, id_min_y


def some_tests(A, compactness=60, kernel_size=11):
    ps = A[0*nb_pts*nb_pts + 3*nb_pts+3, ...]
    ps = ps[ps.nonzero()].reshape(-1,2)

    hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=kernel_size, density=True)
    hist, edges = hist.hist.numpy(), hist.bin_edges
    #hist_np, xedges, yedges = np.histogram2d(ps[...,1], ps[...,0], bins=kernel_size, density=True)

    fig, ax = plt.subplots()
    #ax.hist2d(ps[...,1], ps[...,0], bins=70)
    ax.scatter(ps[...,1], ps[...,0])
    plt.figure()
    #ax.scatter(hist[...,1], hist[...,0])
    plt.imshow(hist.T, origin='lower', interpolation='nearest', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
    ax.axis('equal')
    plt.show()

    time_start = time.time()
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    print(time.time() - time_start)
    print("Base shapes: ", U.shape, S.shape, V.shape)

    sub_U = U[:, :compactness]
    sub_S = S[:compactness]
    sub_V = V[:compactness, :]

    print("Decomp error: ", torch.dist(A, U @ torch.diag(S) @ V))
    print("Compact error: ", torch.dist(A, sub_U @ torch.diag(sub_S) @ sub_V))

    sub_G = sub_U
    sub_W = torch.diag(sub_S) @ sub_V
    print("End shapes: ", sub_G.shape, sub_W.shape)

    plt.figure()
    plt.imshow((sub_G @ sub_W)[0,:].reshape(kernel_size,kernel_size).T, interpolation='nearest')
    plt.show()

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

    return bins_i, bins_j, centroid

def generate_psf_at_pos(pos_x, pos_y, pos_lambda, limit, nb_pts, sub_G, sub_W, kernel_size=11, show=True):

    pos_x = torch.tensor([pos_x])
    pos_y = torch.tensor([pos_y])
    pos_lambda = torch.tensor([pos_lambda])

    id_x = (pos_x+limit)/(2*limit)*(nb_pts-1)
    id_y = (nb_pts-1) - (pos_y+limit)/(2*limit)*(nb_pts-1) # Invert y axis because top left is (0,0) in the PSF matrix and corresponds to +limit in the pattern (-lim -> nbpts ; lim -> 0)

    id_x_close = torch.round((torch.tensor([pos_x])+limit)/(2*limit)*(nb_pts-1), decimals=1)
    id_y_close = torch.round((nb_pts-1) - (torch.tensor([pos_y])+limit)/(2*limit)*(nb_pts-1), decimals=1)

    corner_1 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.floor(id_y_close)*nb_pts + torch.floor(id_x_close)).int(), ...]              # NW

    corner_2 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.ceil(id_y_close)*nb_pts + torch.floor(id_x_close)).int(), ...]               # SW

    corner_3 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.floor(id_y_close)*nb_pts + torch.ceil(id_x_close)).int(), ...]               # NE

    corner_4 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.ceil(id_y_close)*nb_pts + torch.ceil(id_x_close)).int(), ...]                # SE
    
    delta_fx = id_x - torch.floor(id_x_close)
    delta_fy = id_y - torch.floor(id_y_close)
    
    delta_cx = 1 - delta_fx
    delta_cy = 1 - delta_fy

    #weight_1 = (torch.ceil(id_x)-id_x)*(id_y-torch.floor(id_y))
    #weight_2 = (torch.ceil(id_x)-id_x)*(torch.ceil(id_y)-id_y)
    #weight_3 = (id_x - torch.floor(id_x))*(id_y-torch.floor(id_y))
    #weight_4 = (id_x - torch.floor(id_x))*(torch.ceil(id_y)-id_y)

    weight_1 = delta_cx*delta_fy
    weight_2 = delta_cx*delta_cy
    weight_3 = delta_fx*delta_fy
    weight_4 = delta_fx*delta_cy

    w_PSF_at_pos = corner_1*weight_1 + corner_2*weight_2 + corner_3*weight_3 + corner_4*weight_4

    PSF_at_pos = w_PSF_at_pos @ sub_W

    """ print(f"id_x: {id_x}")
    print(f"id_y: {id_y}")
    print(f"w1: {weight_1}")
    print(f"w2: {weight_2}")
    print(f"w3: {weight_3}")
    print(f"w4: {weight_4}") """

    if show:
        plt.figure()
        plt.imshow(PSF_at_pos.reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest')
        plt.show()

    return PSF_at_pos

def generate_psf_at_pos_detector(pos_x, pos_y, pos_lambda, limit, nb_pts, nb_pts_x, nb_pts_y,
                                 id_min_x, id_min_y, sub_G, sub_W, kernel_size=11, show=True):

    pos_x = torch.tensor([pos_x])
    pos_y = torch.tensor([pos_y])
    pos_lambda = torch.tensor([pos_lambda])

    id_x = (pos_x+limit)/(2*limit)*(nb_pts-1) + abs(id_min_x)
    id_y = (nb_pts_y-1) - ((pos_y+limit)/(2*limit)*(nb_pts-1) + abs(id_min_y))  # Invert y axis because top left is (0,0) in the PSF matrix and corresponds to +limit in the pattern (-lim -> nbpts ; lim -> 0)

    id_x_close = torch.round((torch.tensor([pos_x])+limit)/(2*limit)*(nb_pts-1) + abs(id_min_x), decimals=1)
    id_y_close = torch.round((nb_pts_y-1) - ((torch.tensor([pos_y])+limit)/(2*limit)*(nb_pts-1) + abs(id_min_y)), decimals=1)

    """ id_x_close = torch.round((torch.tensor([pos_x])+limit)/(2*limit)*(nb_pts-1) + abs(id_min_x))
    id_y_close = torch.round((nb_pts_y-1) - ((torch.tensor([pos_y])+limit)/(2*limit)*(nb_pts-1) + abs(id_min_y)))
    print(id_x_close, id_y_close) """
    
    """ corner_1 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.floor(id_y_close)*nb_pts_x + torch.floor(id_x_close)).int(), ...]              # NW

    corner_2 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.ceil(id_y_close)*nb_pts_x + torch.floor(id_x_close)).int(), ...]               # SW

    corner_3 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.floor(id_y_close)*nb_pts_x + torch.ceil(id_x_close)).int(), ...]               # NE

    corner_4 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.ceil(id_y_close)*nb_pts_x + torch.ceil(id_x_close)).int(), ...]                # SE """
    corner_1 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.floor(id_y)*nb_pts_x + torch.floor(id_x)).int(), ...]              # NW

    corner_2 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.ceil(id_y)*nb_pts_x + torch.floor(id_x)).int(), ...]               # SW

    corner_3 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.floor(id_y)*nb_pts_x + torch.ceil(id_x)).int(), ...]               # NE

    corner_4 = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.ceil(id_y)*nb_pts_x + torch.ceil(id_x)).int(), ...]                # SE

    #delta_fx = id_x - torch.floor(id_x_close+1e-3)
    #delta_fy = id_y - torch.floor(id_y_close+1e-3)

    delta_fx = id_x - torch.floor(id_x)
    delta_fy = id_y - torch.floor(id_y)
    
    delta_cx = 1 - delta_fx
    delta_cy = 1 - delta_fy

    weight_1 = delta_cx*delta_fy
    weight_2 = delta_cx*delta_cy
    weight_3 = delta_fx*delta_fy
    weight_4 = delta_fx*delta_cy

    if weight_1 < 0 or weight_2 < 0 or weight_3 < 0 or weight_4 < 0:
        print(f"x: {id_x}, {id_x_close}; y: {id_y}, {id_y_close}")
        print(f"Negative weights: {weight_1}, {weight_2}, {weight_3}, {weight_4}")

    """ print(f"w1: {weight_1}")
    print(corner_1)
    print(f"w2: {weight_2}")
    print(corner_2)
    print(f"w3: {weight_3}")
    print(corner_3)
    print(f"w4: {weight_4}")
    print(corner_4)
    print("")
    print(id_x, id_y, pos_lambda)
    print("") """
    w_PSF_at_pos = corner_1*weight_1 + corner_2*weight_2 + corner_3*weight_3 + corner_4*weight_4

    #PSF_at_pos = sub_G[(pos_lambda*nb_pts_x*nb_pts_y + torch.round(id_y_close)*nb_pts_x + torch.round(id_x_close)).int(), ...] @ sub_W
    PSF_at_pos = w_PSF_at_pos @ sub_W

    if show:
        plt.figure()
        plt.imshow(PSF_at_pos.reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest')
        plt.show()

    return PSF_at_pos


def generate_decomp_error_graph(A, U, S, V, min=20, max=122):
    decomp_errors = [0 for i in range(min,max)]
    for i in range(min,max):
        sub_U = U[:, :i]
        sub_S = S[:i]
        sub_V = V[:i, :]

        decomp_errors[i-min] = torch.dist(A, sub_U @ torch.diag(sub_S) @ sub_V)

    plt.figure()
    plt.plot(list(range(min,max)), decomp_errors)
    plt.title("Decomposition error w.r.t. K")
    plt.xlabel("Value of K")
    plt.ylabel("Error")
    plt.show()

def compare_to_A(A, K=None, min=20, max=122, nb_compar=None):
    if (nb_compar is None) or (nb_compar > A.shape[0]):
        nb_compar = A.shape[0]
    
    np.random.seed(0)
    list_compare = np.random.choice(A.shape[0], nb_compar, replace=False)

    U, S, V = torch.linalg.svd(A, full_matrices=False)
    if K is None:
        K = list(range(min, max))
    else:
        K = list(K)
    list_rmse = np.zeros((len(K), nb_compar))
    time_start = time.time()
    for k, compactness in enumerate(K):
        sub_U = U[:, :compactness]
        sub_S = S[:compactness]
        sub_V = V[:compactness, :]

        sub_G = sub_U @ torch.diag(sub_S)
        sub_W = sub_V

        sub_A = (sub_G @ sub_W)

        for i, idx in enumerate(list_compare):
            rmse = torch.dist(A[idx, :], sub_A[idx, :])
            list_rmse[k, i] = rmse
        print(f"Finished for K={compactness} in {time.time() - time_start} seconds")
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + f"rmse_compression_loss_{A.shape[0]//(nb_pts*nb_pts)}w_{limit}lim_{nb_pts}pts.npy", list_rmse)
    return list_rmse

from nmf import run_nmf

def compare_with_database(A, limit, nb_pts, K=None, k_min=20, k_max=122, nb_compar=2000, method = 'svd',
                          sub_folder = "database_for_comparison", pixel_size=0.02, kernel_size=11, show=True):
    if method == 'svd':
        time_svd = time.time()
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        print(f"Finished SVD in {(time.time() - time_svd):.3f} seconds")
    elif method == 'svd_squared':
        U, S, V = torch.linalg.svd(torch.sqrt(A), full_matrices=False)
    list_of_psf = list(glob.glob(dir_name + f"{sub_folder}/*.npy"))

    if K is None:
        K = list(range(k_min, k_max))
    else:
        K = list(K)
    list_rmse = np.zeros((len(K), nb_compar))
    time_start = time.time()


    for k, compactness in enumerate(K):
        if 'svd' in method:
            sub_U = U[:, :compactness]
            sub_S = S[:compactness]
            sub_V = V[:compactness, :]

            sub_G = sub_U @ torch.diag(sub_S)
            sub_W = sub_V
        elif method == 'nmf':
            time_nmf = time.time()
            sub_G, sub_W, err = run_nmf(A, n_components=compactness, init='nndsvdar')
            print(f"Finished NMF for K={compactness} in {(time.time() - time_nmf):.3f} seconds")
            sub_G = torch.from_numpy(sub_G)
            sub_W = torch.from_numpy(sub_W)

        fig, ax = plt.subplots(compactness//6 + 1,6)
        plot_min = torch.min(sub_W).numpy()
        plot_max = torch.max(sub_W).numpy()
        cmap = 'seismic' if 'svd' in method else 'viridis'
        vmin = plot_min
        vmax = plot_max
        vmin = min(vmin, -vmax)
        vmax = -vmin
        for i in range(compactness):
            bar = ax[i // 6, i%6].imshow(sub_W[i,:].reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(bar, cax=cbar_ax)
        plt.show()
        

        #print(f"Negative values in sub_U: {torch.sum(sub_U < 0)}")
        #print(f"Negative values in sub_S: {torch.sum(sub_S < 0)}")
        #print(f"Negative values in sub_V: {torch.sum(sub_V < 0)}")
        #print(f"Negative values in sub_G: {torch.sum(sub_G < 0)}")
        #print(f"Negative values in sub_W: {torch.sum(sub_W < 0)}")

        for i, file_path in enumerate(list_of_psf[:nb_compar]):
            PSF_mat = np.load(file_path)
            parsing = file_path.split('/')[-1].split("_")
            x_pos = float(parsing[1])
            y_pos = float(parsing[3])
            lambda_pos = int(parsing[5][:-4])

            ps = PSF_mat
            ps = np.nan_to_num(ps, nan=0.0)
            ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
            
            #hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=kernel_size, density=False).hist
            bins_i, bins_j, _ = find_bins(ps, pixel_size, kernel_size)
            hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=(bins_j, bins_i), density=False).hist
            hist = hist.T.numpy()
            PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, sub_G, sub_W, kernel_size=kernel_size, show=False)
            #PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, A, torch.eye(kernel_size*kernel_size), kernel_size=kernel_size, show=False)
            if method=='svd_squared':
                PSF_at_pos = torch.square(PSF_at_pos)           #### SECOND METHOD
            PSF_at_pos = PSF_at_pos.reshape(kernel_size,kernel_size).T.numpy()

            rmse = np.sqrt(np.mean((hist - PSF_at_pos)**2))
            list_rmse[k, i] = rmse

            print(f"rmse: {rmse:.2f}")
            ind_max = np.argmax(np.abs(hist-PSF_at_pos))
            print(f"max err: {np.abs(hist-PSF_at_pos).flatten()[ind_max]:.2f}")
            print(f"max err %: {(np.abs(hist-PSF_at_pos)/hist).flatten()[ind_max]*100:.1f}%")
            print(f"hist val: {hist.flatten()[ind_max]:.2f}")
            print(f"recons val: {PSF_at_pos.flatten()[ind_max]:.2f}")
            print("")

            if show:
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(hist, origin='lower', interpolation='nearest', vmin=min(hist.min(), PSF_at_pos.min()), vmax=max(hist.max(), PSF_at_pos.max()))
                ax[0].set_title("Original PSF")
                bar = ax[1].imshow(PSF_at_pos, origin='lower', interpolation='nearest', vmin=min(hist.min(), PSF_at_pos.min()), vmax=max(hist.max(), PSF_at_pos.max()))
                ax[1].set_title("Reconstructed PSF")
                print(f"Position: x={x_pos:.2f}mm, y={y_pos:.2f}mm at w={np.linspace(450,650,55)[lambda_pos]:.2f}nm")
                print(f"RMSE for K={compactness}: {rmse}")
                fig.suptitle(f"Position: x={x_pos:.2f}mm, y={y_pos:.2f}mm at w={np.linspace(450,650,55)[lambda_pos]:.2f}nm")
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(bar, cax=cbar_ax)
                plt.show()
                

        print(f"Finished for K={compactness} in {time.time() - time_start} seconds")
    if 'svd' in method:
        prefix = "rmse"
    elif method == 'nmf':
        prefix = "rmse_nmf"

    file_name = f"{prefix}_nmf_comparison_{A.shape[0]//(nb_pts*nb_pts)}w_{limit}lim_{nb_pts}pts.npy"
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + file_name, list_rmse)
    return list_rmse

def heatmap_from_database(A, limit, nb_pts, K=None, k_min=20, k_max=122, nb_compar=2000,
                          pixel_size=0.02, kernel_size=11, normalize='same'):
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    list_of_psf = list(glob.glob(dir_name + "database_for_comparison/*.npy"))

    if K is None:
        K = list(range(k_min, k_max))
    else:
        K = list(K)
    heatmap_setup = np.zeros((len(K), nb_pts, nb_pts, int(np.floor(nb_compar/(nb_pts*nb_pts)*10))))
    time_start = time.time()
    for k, compactness in enumerate(K):
        sub_U = U[:, :compactness]
        sub_S = S[:compactness]
        sub_V = V[:compactness, :]

        sub_G = sub_U @ torch.diag(sub_S)
        sub_W = sub_V
        for i, file_path in enumerate(list_of_psf[:nb_compar]):
            PSF_mat = np.load(file_path)
            parsing = file_path.split('/')[-1].split("_")
            x_pos = float(parsing[1])
            y_pos = float(parsing[3])
            lambda_pos = int(parsing[5][:-4])

            id_x = torch.floor((torch.tensor([x_pos])+limit)/(2*limit)*nb_pts).int()
            id_y = torch.floor(nb_pts - (torch.tensor([y_pos])+limit)/(2*limit)*nb_pts).int()

            ps = PSF_mat
            ps = np.nan_to_num(ps, nan=0.0)
            ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)

            bins_i, bins_j, _ = find_bins(ps, pixel_size, kernel_size)
            hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=(bins_j, bins_i), density=False).hist
            hist = hist.T.numpy()

            PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, sub_G, sub_W, kernel_size=kernel_size, show=False)
            PSF_at_pos = PSF_at_pos.reshape(kernel_size,kernel_size).T.numpy()

            if normalize=='same':
                hist = hist/ps.shape[0]
                PSF_at_pos = PSF_at_pos/ps.shape[0]
            elif normalize=='indep':
                hist = hist/np.sum(hist)
                PSF_at_pos = PSF_at_pos/np.sum(PSF_at_pos)

            rmse = np.sqrt(np.mean((hist - PSF_at_pos)**2))

            ind = np.where(heatmap_setup[k, id_y, id_x, :]==0)[0][0]
            heatmap_setup[k, id_y, id_x, ind] = rmse

        print(f"Finished for K={compactness} in {time.time() - time_start} seconds")
    if normalize in ['same', 'indep']:
        file_name = f"rmse_heatmap_normalized_{normalize}_{A.shape[0]//(nb_pts*nb_pts)}w_{limit}lim_{nb_pts}pts.npy"
    else:
        file_name = f"rmse_heatmap_{A.shape[0]//(nb_pts*nb_pts)}w_{limit}lim_{nb_pts}pts.npy"
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + file_name, heatmap_setup)
    return heatmap_setup

def compute_graph_heatmap_from_database(list_of_files, limit, nb_pts, n_lam = 55, sub_folder = "database_for_comparison", K=None, k_min=20, k_max=122, nb_compar=2000,
                                        pos_on_det = False,
                                        pixel_size=0.02, kernel_size=11, method = 'svd'):
    if pos_on_det:
        A, nb_pts_x, nb_pts_y, id_min_x, id_min_y = compute_A_pos_detector(list_of_files, nb_pts, limit, n_lam, pixel_size= pixel_size, kernel_size=kernel_size)
    else:
        A = compute_A(list_of_files, nb_pts, n_lam, pixel_size= pixel_size, kernel_size=kernel_size)

    if method == 'svd':
        U, S, V = torch.linalg.svd(A, full_matrices=False)
    elif method == 'svd_squared':
        U, S, V = torch.linalg.svd(torch.sqrt(A), full_matrices=False)
    list_of_psf = list(glob.glob(dir_name + f"{sub_folder}/*.npy"))

    if K is None:
        K = list(range(k_min, k_max))
    else:
        K = list(K)
    list_rmse = np.zeros((len(K),min(nb_compar, len(list_of_psf))))
    time_start = time.time()

    heatmap_setup_indep = np.zeros((len(K), nb_pts, nb_pts, n_lam, 5)) - 1
    heatmap_setup_same = np.zeros((len(K), nb_pts, nb_pts, n_lam, 5)) - 1
    heatmap_setup_false = np.zeros((len(K), nb_pts, nb_pts, n_lam, 5)) - 1

    for k, compactness in enumerate(K):
        if 'svd' in method:
            sub_U = U[:, :compactness]
            sub_S = S[:compactness]
            sub_V = V[:compactness, :]

            sub_G = sub_U @ torch.diag(sub_S)
            sub_W = sub_V
        elif method == 'nmf':
            sub_G, sub_W, err = run_nmf(A, n_components=compactness)
            sub_G = torch.from_numpy(sub_G)
            sub_W = torch.from_numpy(sub_W)

        """ fig, ax = plt.subplots(compactness//6 + 1,6)
        plot_min = torch.min(sub_W).numpy()
        plot_max = torch.max(sub_W).numpy()
        cmap = 'seismic' if 'svd' in method else 'viridis'
        vmin = plot_min
        vmax = plot_max
        vmin = min(vmin, -vmax)
        vmax = -vmin
        for i in range(compactness):
            bar = ax[i // 6, i%6].imshow(sub_W[i,:].reshape(kernel_size,kernel_size).T, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(bar, cax=cbar_ax)
        plt.show() """


        for i, file_path in enumerate(list_of_psf[:nb_compar]):
            PSF_mat = np.load(file_path)
            parsing = file_path.split('/')[-1].split("_")
            x_pos = float(parsing[1])
            y_pos = float(parsing[3])
            lambda_pos = int(parsing[5][:-4])

            ps = PSF_mat
            ps = np.nan_to_num(ps, nan=0.0)
            ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
            

            #hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=kernel_size, density=False).hist
            bins_i, bins_j, centroid = find_bins(ps, pixel_size, kernel_size)
            hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=(bins_j, bins_i), density=False).hist
            hist = hist.T.numpy()
            if pos_on_det:
                PSF_at_pos = generate_psf_at_pos_detector(centroid[1], centroid[0], lambda_pos, limit, nb_pts, nb_pts_x, nb_pts_y,
                                                          id_min_x, id_min_y, sub_G, sub_W, kernel_size=kernel_size, show=False)
                #PSF_at_pos = generate_psf_at_pos_detector(centroid[1], centroid[0], lambda_pos, limit, nb_pts, nb_pts_x, nb_pts_y, id_min_x, id_min_y, A, torch.eye(kernel_size*kernel_size), kernel_size=kernel_size, show=False)
            else:
                PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, sub_G, sub_W, kernel_size=kernel_size, show=False)
                #PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, A, torch.eye(kernel_size*kernel_size), kernel_size=kernel_size, show=False)
            if method=='svd_squared':
                PSF_at_pos = torch.square(PSF_at_pos)           #### SECOND METHOD
            neg_sum = - torch.sum(PSF_at_pos[PSF_at_pos < 0])
            abs_sum = torch.sum(torch.abs(PSF_at_pos))
            PSF_at_pos = torch.clamp(PSF_at_pos, min=0.0)
            PSF_at_pos = PSF_at_pos.reshape(kernel_size,kernel_size).T.numpy()

            mask = np.logical_or(hist > 0, PSF_at_pos > 0)
            rmse_mask = np.sqrt(np.mean(((hist - PSF_at_pos)[mask])**2))
            rmse = np.sqrt(np.mean(((hist - PSF_at_pos))**2))
            list_rmse[k, i] = rmse
            
            #if (abs(y_pos) <= 1e-3) and (abs(x_pos) <= 1e-3):
            if rmse > 550:
                print(f"rmse prev: {rmse:.2f}")
                print(f"rmse: {rmse_mask:.2f}")
                print(y_pos)
                print(x_pos)
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(hist, origin='lower', interpolation='nearest', vmin=min(hist.min(), PSF_at_pos.min()), vmax=max(hist.max(), PSF_at_pos.max()))
                ax[0].set_title("Original PSF")
                bar = ax[1].imshow(PSF_at_pos, origin='lower', interpolation='nearest', vmin=min(hist.min(), PSF_at_pos.min()), vmax=max(hist.max(), PSF_at_pos.max()))
                ax[1].set_title("Reconstructed PSF")
                print(f"Position: x={x_pos:.2f}mm, y={y_pos:.2f}mm at w={np.linspace(450,650,55)[lambda_pos]:.2f}nm")
                print(f"RMSE for K={compactness}: {rmse}")
                fig.suptitle(f"Position: x={x_pos:.2f}mm, y={y_pos:.2f}mm at w={np.linspace(450,650,55)[lambda_pos]:.2f}nm")
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(bar, cax=cbar_ax)
                plt.show()

            """ print(f"rmse: {rmse:.2f}")
            ind_max = np.argmax(np.abs(hist-PSF_at_pos))
            print(f"Proportion negative: {neg_sum/abs_sum*100}%")
            print(f"max err: {np.abs(hist-PSF_at_pos).flatten()[ind_max]:.2f}")
            print(f"max err %: {(np.abs(hist-PSF_at_pos)/hist).flatten()[ind_max]*100:.1f}%")
            print(f"hist val: {hist.flatten()[ind_max]:.2f}")
            print(f"recons val: {PSF_at_pos.flatten()[ind_max]:.2f}")
            print("") """

            hist_same = hist/ps.shape[0]
            PSF_at_pos_same = PSF_at_pos/ps.shape[0]

            hist = hist/np.sum(hist)
            PSF_at_pos_indep = PSF_at_pos/np.sum(PSF_at_pos)

            rmse_same = np.sqrt(np.mean((hist_same - PSF_at_pos_same)**2))
            rmse_indep = np.sqrt(np.mean((hist_same - PSF_at_pos_indep)**2))

            #id_x = torch.floor((torch.tensor([x_pos])+limit)/(2*limit)*(nb_pts-1)).int()
            #id_y = torch.floor((nb_pts-1) - (torch.tensor([y_pos])+limit)/(2*limit)*(nb_pts-1)).int()

            id_x = torch.round((torch.tensor([x_pos])+limit)/(2*limit)*(nb_pts-1), decimals=1).int()
            id_y = torch.round((nb_pts-1) - (torch.tensor([y_pos])+limit)/(2*limit)*(nb_pts-1), decimals=1).int()
            #id_y = torch.round((torch.tensor([y_pos])+limit)/(2*limit)*(nb_pts-1), decimals=1).int()
            """ if (lambda_pos == 19) and (id_y == 4) and (rmse > 5):
                print("Position: ", x_pos, y_pos)
                plt.scatter(ps[...,1], ps[...,0], s=0.1)
                plt.title(f"Position: x={id_x}")
                plt.show() """

            ind_same = np.where(heatmap_setup_same[k, id_y, id_x, lambda_pos, :]==-1)[0][0]
            ind_indep = np.where(heatmap_setup_indep[k, id_y, id_x, lambda_pos, :]==-1)[0][0]
            ind_false = np.where(heatmap_setup_false[k, id_y, id_x, lambda_pos, :]==-1)[0][0]
            
            heatmap_setup_same[k, id_y, id_x, lambda_pos, ind_same] = rmse_same
            heatmap_setup_indep[k, id_y, id_x, lambda_pos, ind_indep] = rmse_indep
            heatmap_setup_false[k, id_y, id_x, lambda_pos, ind_false] = rmse
                

        print(f"Finished for K={compactness} in {time.time() - time_start:.3f} seconds")
        print(f"Mean unnormalized RMSE: {np.mean(list_rmse[k, :]):.3f}")
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
    return list_rmse

def plot_rmse(limit, nb_pts, w=55, file_path = "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/", data='comparison', method = 'svd', grid_type = "regular", measure='mean'):
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
    #plt.plot(list(range(122-rmse_list.shape[0],122)), rmse_list)
    plt.plot(list(range(3,rmse_list.shape[0]+3)), rmse_list)
    plt.title(f"RMSE w.r.t. K for {measure} value")
    plt.xlabel("Value of K")
    plt.ylabel("RMSE")
    plt.show()

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
#pixel_size = 0.02

#A = compute_A(list_of_files, nb_pts, n_lam, pixel_size= pixel_size, kernel_size=kernel_size)
#A, _ = compute_A_pos_detector(list_of_files, nb_pts, limit, n_lam, pixel_size= pixel_size, kernel_size=kernel_size)
#compare_to_A(A, K=list(range(10,122)), nb_compar=1000)
compute_graph_heatmap_from_database(list_of_files, limit, nb_pts, n_lam, sub_folder = "2.56_exact_regular_database_for_comparison", K=list(range(40,41)), method = 'svd', nb_compar=4000,
                                    pos_on_det=False, pixel_size=pixel_size, kernel_size=kernel_size)
#compare_with_database(A, limit, nb_pts, K=list(range(3, 122)), nb_compar=4000, method = 'svd', sub_folder = "2.56_exact_regular_database_for_comparison", pixel_size=pixel_size, kernel_size=kernel_size, show=True)
#heatmap_from_database(A, limit, nb_pts, K=list(range(10,122)), nb_compar=4000, pixel_size=pixel_size, kernel_size=kernel_size, normalize='indep')
#heatmap_from_database(A, limit, nb_pts, K=list(range(10,122)), nb_compar=4000, pixel_size=pixel_size, kernel_size=kernel_size, normalize='same')
#heatmap_from_database(A, limit, nb_pts, K=list(range(10,122)), nb_compar=4000, pixel_size=pixel_size, kernel_size=kernel_size, normalize=False)
#plot_rmse(limit, nb_pts, w=n_lam, data='compression', measure='mean')
#plot_rmse(limit, nb_pts, w=n_lam, data='comparison', method = 'svd', grid_type = "regular", measure='mean')
#plot_heatmap(limit, nb_pts, K=0, w=n_lam, measure='mean', method='svd', grid_type = "regular", normalize = 'indep')
#plot_heatmap(limit, nb_pts, K=0, w=n_lam, measure='mean', method='svd', grid_type = "regular", normalize = 'same')
plot_heatmap(limit, nb_pts, K=0, w=n_lam, measure='mean', method='svd', grid_type = "regular", normalize = False)