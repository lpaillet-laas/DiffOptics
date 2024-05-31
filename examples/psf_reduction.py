import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import glob

limit = 8
nb_pts = 40

dir_name = "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/psfs/"
file_name = f"psf_field_w_520.0_lim_{limit}_pts_{nb_pts}.npy"


list_of_files = list(glob.glob(dir_name + f"psf_field_*_lim_{limit}_pts_{nb_pts}.npy"))
list_of_files.sort()
n_lam = len(list_of_files)

def compute_A(list_of_files, nb_pts, n_lam):
    A = torch.zeros((nb_pts*nb_pts*n_lam, 11*11))

    for f, file_path in enumerate(list_of_files):
        PSF_mat = np.load(file_path) # (nb_pts, nb_pts, <10000, 2)

        PSF_mat = np.flip(PSF_mat, (0,))
        PSF_mat = np.transpose(PSF_mat, (1, 0, 2, 3))
        for i in range(PSF_mat.shape[0]):
            for j in range(PSF_mat.shape[1]):
                ps = PSF_mat[i, j, ...]
                ps = np.nan_to_num(ps, nan=0)
                ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=11, density=False).hist
                A[f*nb_pts*nb_pts + i*PSF_mat.shape[1] + j, :] = hist.flatten()

    #A = A[~torch.any(A.isnan(), dim=1), :]
    A = torch.nan_to_num(A, nan=0.0)
    return A

def different_tests(A, compactness=60):
    ps = A[0*nb_pts*nb_pts + 3*nb_pts+3, ...]
    ps = ps[ps.nonzero()].reshape(-1,2)

    hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=11, density=True)
    hist, edges = hist.hist.numpy(), hist.bin_edges
    #hist_np, xedges, yedges = np.histogram2d(ps[...,1], ps[...,0], bins=11, density=True)

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
    plt.imshow((sub_G @ sub_W)[0,:].reshape(11,11).T, interpolation='nearest')
    plt.show()

def generate_psf_at_pos(pos_x, pos_y, pos_lambda, limit, nb_pts, sub_G, sub_W, show=True):

    pos_x = torch.tensor([pos_x])
    pos_y = torch.tensor([pos_y])
    pos_lambda = torch.tensor([pos_lambda])

    id_x = (pos_x+limit)/(2*limit)*nb_pts
    id_y = nb_pts - (pos_y+limit)/(2*limit)*nb_pts # Invert y axis because top left is (0,0) in the PSF matrix and corresponds to +limit in the pattern (-lim -> nbpts ; lim -> 0)

    #corner_1 = PSF_mat[torch.floor(id_x), torch.floor(id_y), ...]
    corner_1 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.floor(id_y)*nb_pts + torch.floor(id_x)).int(), ...]
    weight_1 = torch.sqrt(torch.square(1-(id_x - torch.floor(id_x))) + torch.square(1-(id_y - torch.floor(id_y))))

    corner_2 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.ceil(id_y)*nb_pts + torch.floor(id_x)).int(), ...]
    weight_2 = torch.sqrt(torch.square(1-(id_x - torch.floor(id_x))) + torch.square(1-(id_y - torch.ceil(id_y))))

    corner_3 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.floor(id_y)*nb_pts + torch.ceil(id_x)).int(), ...]
    weight_3 = torch.sqrt(torch.square(1-(id_x - torch.ceil(id_x))) + torch.square(1-(id_y - torch.floor(id_y))))

    corner_4 = sub_G[(pos_lambda*nb_pts*nb_pts + torch.ceil(id_y)*nb_pts + torch.ceil(id_x)).int(), ...]
    weight_4 = torch.sqrt(torch.square(1-(id_x - torch.ceil(id_x))) + torch.square(1-(id_y - torch.ceil(id_y))))

    w_PSF_at_pos = (corner_1*weight_1 + corner_2*weight_2 + corner_3*weight_3 + corner_4*weight_4)/(weight_1 + weight_2 + weight_3 + weight_4)
    PSF_at_pos = w_PSF_at_pos @ sub_W

    if show:
        plt.figure()
        plt.imshow(PSF_at_pos.reshape(11,11).T, origin='lower', interpolation='nearest')
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

def compare_with_database(A, limit, nb_pts, K=None, min=20, max=122, nb_compar=2000, show=True):
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    list_of_psf = list(glob.glob(dir_name + "database_for_comparison/*.npy"))

    if K is None:
        K = list(range(min, max))
    else:
        K = list(K)
    list_rmse = np.zeros((len(K), nb_compar))
    time_start = time.time()
    for k, compactness in enumerate(K):
        for i, file_path in enumerate(list_of_psf[:nb_compar]):
            PSF_mat = np.load(file_path)
            parsing = file_path.split('/')[-1].split("_")
            x_pos = float(parsing[1])
            y_pos = float(parsing[3])
            lambda_pos = int(parsing[5][:-4])

            ps = PSF_mat
            ps = np.nan_to_num(ps, nan=0.0)
            ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
            hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=11, density=False).hist
            hist = hist.T.numpy()

            sub_U = U[:, :compactness]
            sub_S = S[:compactness]
            sub_V = V[:compactness, :]

            sub_G = sub_U
            sub_W = torch.diag(sub_S) @ sub_V

            PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, sub_G, sub_W, show=False)
            PSF_at_pos = PSF_at_pos.reshape(11,11).T.numpy()

            rmse = np.sqrt(np.mean((hist - PSF_at_pos)**2))
            list_rmse[k, i] = rmse

            if show:
                plt.figure()
                plt.imshow(hist, origin='lower', interpolation='nearest')
                plt.figure()
                plt.imshow(PSF_at_pos, origin='lower', interpolation='nearest')
                plt.show()
                print(f"RMSE for K={compactness}: {rmse}")

        print(f"Finished for K={compactness} in {time.time() - time_start} seconds")
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + f"rmse_comparison_{A.shape[0]//(nb_pts*nb_pts)}w_{limit}lim_{nb_pts}pts.npy", list_rmse)
    return list_rmse

def heatmap_from_database(A, limit, nb_pts, K=None, min=20, max=122, nb_compar=2000, normalize=True):
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    list_of_psf = list(glob.glob(dir_name + "database_for_comparison/*.npy"))

    if K is None:
        K = list(range(min, max))
    else:
        K = list(K)
    heatmap_setup = np.zeros((len(K), nb_pts, nb_pts, int(np.floor(nb_compar/(nb_pts*nb_pts)*10))))
    time_start = time.time()
    for k, compactness in enumerate(K):
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
            hist = torch.histogramdd(torch.from_numpy(ps).flip(1), bins=11, density=False).hist
            hist = hist.T.numpy()

            sub_U = U[:, :compactness]
            sub_S = S[:compactness]
            sub_V = V[:compactness, :]

            sub_G = sub_U
            sub_W = torch.diag(sub_S) @ sub_V

            PSF_at_pos = generate_psf_at_pos(x_pos, y_pos, lambda_pos, limit, nb_pts, sub_G, sub_W, show=False)
            PSF_at_pos = PSF_at_pos.reshape(11,11).T.numpy()

            if normalize:
                hist = hist/ps.shape[0]
                PSF_at_pos = PSF_at_pos/ps.shape[0]

            rmse = np.sqrt(np.mean((hist - PSF_at_pos)**2))

            ind = np.where(heatmap_setup[k, id_y, id_x, :]==0)[0][0]
            heatmap_setup[k, id_y, id_x, ind] = rmse

        print(f"Finished for K={compactness} in {time.time() - time_start} seconds")
    if normalize:
        file_name = f"rmse_heatmap_normalized_{A.shape[0]//(nb_pts*nb_pts)}w_{limit}lim_{nb_pts}pts.npy"
    else:
        file_name = f"rmse_heatmap_{A.shape[0]//(nb_pts*nb_pts)}w_{limit}lim_{nb_pts}pts.npy"
    np.save("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/" + file_name, heatmap_setup)
    return heatmap_setup

def plot_rmse(limit, nb_pts, w=55, file_path = "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/", measure='mean'):
    rmse_list = np.load(file_path + f"rmse_comparison_{w}w_{limit}lim_{nb_pts}pts.npy")
    if measure == 'mean':
        rmse_list = np.mean(rmse_list, axis=1)
    elif measure == 'median':
        rmse_list = np.median(rmse_list, axis=1)
    elif measure == 'max':
        rmse_list = np.max(rmse_list, axis=1)
    elif measure == 'min':
        rmse_list = np.min(rmse_list, axis=1)
    plt.figure()
    plt.plot(list(range(122-rmse_list.shape[0],122)), rmse_list)
    plt.title(f"RMSE w.r.t. K for {measure} value")
    plt.xlabel("Value of K")
    plt.ylabel("RMSE")
    plt.show()

def plot_heatmap(limit, nb_pts, K=60, w=55, file_path = "/home/lpaillet/Documents/Codes/DiffOptics/examples/render_pattern_demo/", measure='mean', normalize=True):
    if normalize:
        file_name = f"rmse_heatmap_normalized_{w}w_{limit}lim_{nb_pts}pts.npy"
    else:
        file_name = f"rmse_heatmap_{w}w_{limit}lim_{nb_pts}pts.npy"
    heatmap = np.load(file_path + file_name)
    heatmap[heatmap == 0] = np.nan
    if measure == 'mean':
        heatmap = np.nanmean(heatmap, axis=3)
    elif measure == 'median':
        heatmap = np.nanmedian(heatmap, axis=3)
    elif measure == 'max':
        heatmap = np.nanmax(heatmap, axis=3)
    elif measure == 'min':
        heatmap = np.nanmin(heatmap, axis=3)
    
    plt.figure()
    plt.imshow(heatmap[K, ...], origin='lower', interpolation='nearest', extent=[-limit, limit, -limit, limit])
    plt.title(f"RMSE for K={K} and {measure} value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

A = compute_A(list_of_files, nb_pts, n_lam)
#compare_with_database(A, limit, nb_pts, K=list(range(10,122)), nb_compar=4000, show=False)
heatmap_from_database(A, limit, nb_pts, K=list(range(10,122)), nb_compar=4000, normalize=True)
heatmap_from_database(A, limit, nb_pts, K=list(range(10,122)), nb_compar=4000, normalize=False)
#plot_rmse(limit, nb_pts, w=55, measure='mean')
plot_heatmap(limit, nb_pts, K=111, w=55, measure='mean', normalize = True)
plot_heatmap(limit, nb_pts, K=111, w=55, measure='mean', normalize = False)