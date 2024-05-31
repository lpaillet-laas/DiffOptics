import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import diffoptics as do

import time

# initialize a lens
device = torch.device('cpu')
lens = do.Lensgroup(device=device, theta_y = [0], origin=np.array([0,0,0]), shift=np.array([0,0,0]))

save_dir = './autodiff_demo/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

R = 12.7
d_lens_1 = 0.0
d_lens_2 = 6.5
surfaces = [
    do.Aspheric(R, d_lens_1, c=0.037, device=device),
    do.Aspheric(R, d_lens_2, c=0., device=device),
]
materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]

### Test Prisms ###
R = 12.5
d = 120.0
angle = 53.4/2
x1 = R
#x2 = 54
#H = ((x1+x2)/2 - x1)*np.tan(angle*np.pi/180)
H = R*2
x2 = x1 + 2*H*np.tan(angle*np.pi/180)

""" coefs = torch.tensor([[x1, 0, 0],
                      [x1/np.tan(angle*np.pi/180), 1, -1/np.tan(angle*np.pi/180)],
                      [H*np.tan(angle*np.pi/180)+x1, 1, 1/np.tan(angle*np.pi/180)], 
                      [x2, 0, 0]]).float()
coefs = torch.tensor([[-20,0,0], [1,1,1/10]]).float()
coefs = torch.tensor([[x1-11, 0, 0],
                      [x1*np.tan(angle*np.pi/180), 1, np.tan(angle*np.pi/180)],
                      [H*np.tan(angle*np.pi/180)+x1, 1, -np.tan(angle*np.pi/180)], 
                      [x2-36, 0, 0]]).float() """

# coefs = torch.tensor([[x1, 0, 0],
#                       [x1*np.tan(angle*np.pi/180)+R, 0, np.tan(angle*np.pi/180)],
#                       [(H+R)*np.tan(angle*np.pi/180)+x1, 0, -np.tan(angle*np.pi/180)], 
#                       [x2, 0, 0]]).float()

tc = 4.1
te = 2
e = tc-te
x0=x1

F = 75

prism_length = 2*H*np.tan(angle*np.pi/180)
# coefs = torch.tensor([[x0-te-e*2, 0, 0, 0, 0, 0],
#                       [x0-e, 0, 0, 0, 0, -(R+e-x0)/(x0*x0)],
#                       [x0+te, 0, 0, 0, 0, 0],
#                       [x0-e, 0, 0, 0, 0, (R+e-x0)/(x0*x0)]]).float()

d = 2*F - prism_length/2

coefs = torch.tensor([[0, 0, 0, 0, 0, 0],
                      [x0+e, 0, 0, -(R+e-x0)/(x0*x0), 0, -(R+e-x0)/(x0*x0)],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],
                      [0, 0, -np.tan(angle*np.pi/180), 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0],
                      [x0-e, 0, 0, (R+e-x0)/(x0*x0), 0, (R+e-x0)/(x0*x0)],
                      [0, 0, 0, 0, 0, 0]]).float()

""" surfaces = [
    do.XYPolynomial(R, d, J=1, ai=coefs[0], device = device),
    do.XYPolynomial(R, d, J=1, ai=coefs[1], device = device),
    do.XYPolynomial(R, d, J=1, ai=coefs[2], device = device),
    do.XYPolynomial(R, d, J=1, ai=coefs[3], device = device),
] """

""" surfaces = [
    do.XYPolynomial(R, d, J=1, ai=coefs[0][:3], device = device),
    do.XYPolynomial(R, d, J=2, ai=coefs[1], device = device),
    do.XYPolynomial(R, d, J=1, ai=coefs[2][:3], device = device),
    do.XYPolynomial(R, d, J=2, ai=coefs[3], device = device),
] """

surfaces = [
    do.XYPolynomial(R, (d-F+H*np.tan(angle*np.pi/180))-te, J=1, ai=coefs[0][:3], device = device),
    do.XYPolynomial(R, (d-F+H*np.tan(angle*np.pi/180))-x0, J=2, ai=coefs[1], device = device),
    do.XYPolynomial(R, d, J=1, ai=coefs[2][:3], device = device),
    do.XYPolynomial(R, d+R*np.tan(angle*np.pi/180), J=1, ai=coefs[3][:3], device = device),
    do.XYPolynomial(R, d+(H+R)*np.tan(angle*np.pi/180), J=1, ai=coefs[4][:3], device = device),
    do.XYPolynomial(R, d+2*H*np.tan(angle*np.pi/180), J=1, ai=coefs[5][:3], device = device),
    do.XYPolynomial(R, (d+prism_length+F-H*np.tan(angle*np.pi/180))-x0, J=2, ai=coefs[6], device = device),
    do.XYPolynomial(R, (d+prism_length+F-H*np.tan(angle*np.pi/180))+te, J=1, ai=coefs[7][:3], device = device),
]

materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air'),
    do.Material('N-SK2'),
    do.Material('N-SF4'),
    do.Material('N-SK2'),
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]

##################################################################################################################################################
offset_collim = 2.4

# Coefs for the surfaces
coefs = torch.tensor([[0, 0, 0, 0, 0, 0],                                              # Start of lens 1
                      [x0+e, 0, 0, -(R+e-x0)/(x0*x0), 0, -(R+e-x0)/(x0*x0)],           # End of lens 1
                      [0, 0, 0, 0, 0, 0],                                              # Prism 1
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],                        # Prism 2
                      [0, 0, -np.tan(angle*np.pi/180), 0, 0, 0],                       # Prism 3
                      [0, 0, 0, 0, 0, 0],                                              # End of prism
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],                           # Ghost surface
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
#    do.XYPolynomial(R, d + 15 + 2*H*np.tan(angle*np.pi/180), J=1, ai=coefs[6][:3], device = device),                    # Ghost surface
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
    #do.Material('N-BK7'),
    #do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]

###################################################################################################################################################

""" x1 = 1.767909
x2 = 4.531195
x3 = 8.651783
x1_e = x1
x2_e = 7-x2
x3_e = 9.5-x3
R = 12.7
x0 = R
coefs = torch.tensor([[x0 - x1_e, 0, 0, (R+x1_e-x0)/(x0*x0), 0, (R+x1_e-x0)/(x0*x0)],
                      [x0 + x2_e, 0, 0, -(R+x2_e-x0)/(x0*x0), 0, -(R+x2_e-x0)/(x0*x0)],
                      [x0 + x3_e, 0, 0, -(R+x3_e-x0)/(x0*x0), 0, -(R+x3_e-x0)/(x0*x0)]]).float()

surfaces = [
    do.XYPolynomial(R, x1-x0, J=2, ai=coefs[0], device = device),
    do.XYPolynomial(R, x2-x0, J=2, ai=coefs[1], device = device),
    do.XYPolynomial(R, x3-x0, J=2, ai=coefs[2], device = device),
] """


R = 12.7
x1 = 1.767909
x2 = 4.531195
x3 = 8.651783
x1_e = x1
x2_e = 7-x2
x3_e = 9.5-x3
doublet_length = 9.5
x0 = R

prism_length = 2*H*np.tan(angle*np.pi/180)
H = R*2

F = 75

d = 2*F - prism_length/2

doublet_coefs = torch.tensor([[x0 - x1_e, 0, 0, (R+x1_e-x0)/(x0*x0), 0, (R+x1_e-x0)/(x0*x0)],      # Start of doublet 1
                      [x0 + x2_e, 0, 0, -(R+x2_e-x0)/(x0*x0), 0, -(R+x2_e-x0)/(x0*x0)],            # Middle of doublet 1
                      [x0 + x3_e, 0, 0, -(R+x3_e-x0)/(x0*x0), 0, -(R+x3_e-x0)/(x0*x0)],            # End of doublet 1
                      [0, 0, 0, 0, 0, 0],                                                          # Prism 1
                      [0, 0, np.tan(angle*np.pi/180), 0, 0, 0],                                    # Prism 2
                      [0, 0, -np.tan(angle*np.pi/180), 0, 0, 0],                                   # Prism 3
                      [0, 0, 0, 0, 0, 0],                                                          # End of prism
                      [x0 - x3_e, 0, 0, (R+x3_e-x0)/(x0*x0), 0, (R+x3_e-x0)/(x0*x0)],              # Start of doublet 2
                      [x0 - x2_e, 0, 0, (R+x2_e-x0)/(x0*x0), 0, (R+x2_e-x0)/(x0*x0)],              # Middle of doublet 2
                      [x0 + x1_e, 0, 0, -(R+x1_e-x0)/(x0*x0), 0, -(R+x1_e-x0)/(x0*x0)]]).float()   # End of doublet 2



doublet_surfaces = [
    do.XYPolynomial(R, F - doublet_length/2 + x1-x0, J=2, ai=doublet_coefs[0], device = device),              # Start of doublet 1                                                     # Start of doublet 1
    do.XYPolynomial(R, F - doublet_length/2 + x2-x0, J=2, ai=doublet_coefs[1], device = device),              # Middle of doublet 1                                                     # Middle of doublet 1
    do.XYPolynomial(R, F - doublet_length/2 + x3-x0, J=2, ai=doublet_coefs[2], device = device),              # End of doublet 1                                                     # End of doublet 1
    do.XYPolynomial(R, 2*F - prism_length/2, J=1, ai=doublet_coefs[3][:3], device = device),                                                                   # Prism 1
    do.XYPolynomial(R, 2*F - prism_length/2 + R*np.tan(angle*np.pi/180), J=1, ai=doublet_coefs[4][:3], device = device),                                       # Prism 2
    do.XYPolynomial(R, 2*F - prism_length/2 + (H+R)*np.tan(angle*np.pi/180), J=1, ai=doublet_coefs[5][:3], device = device),                                   # Prism 3
    do.XYPolynomial(R, 2*F + prism_length/2, J=1, ai=doublet_coefs[6][:3], device = device),                                     # End of prism
    do.XYPolynomial(R, 3*F + doublet_length/2 -x3-x0, J=2, ai=doublet_coefs[7], device = device),  # Start of doublet 2                                                    # Start of doublet 2
    do.XYPolynomial(R, 3*F + doublet_length/2 -x2-x0, J=2, ai=doublet_coefs[8], device = device),  # Middle of doublet 2                                                    # Middle of doublet 2
    do.XYPolynomial(R, 3*F + doublet_length/2 -x1-x0, J=2, ai=doublet_coefs[9], device = device),  # End of doublet 2                                                    # End of doublet 2
]


doublet_materials = [
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

################################################################################################################################################

doublet_lens = do.Lensgroup(device=device)
doublet_lens.load(doublet_surfaces, doublet_materials)
doublet_lens.d_sensor = 4*F
doublet_lens.r_last = R

doublet_lens.film_size = [131, 131] # [pixels]
doublet_lens.pixel_size = 80.0e-3 # [mm]


lens.load(surfaces, materials)
lens.d_sensor = 4*F + 2*offset_collim # distance to sensor (in mm?)
lens.r_last = R # radius of sensor (in mm?) in plot

lens.film_size = [131, 131] # [pixels]
lens.pixel_size = 80.0e-3 # [mm]

#lens = doublet_lens

# generate array of rays
wavelength = torch.Tensor([550]).to(device) # [nm]
R = 12.7 # [mm]
#M = 131
M = 5
R = 80e-3*(M-1)/2 # [mm*pixel_size]
R = 8

def render():
    ray_init = lens.sample_ray(wavelength, M=M, R=R, sampling='grid')
    ps = lens.trace_to_sensor(ray_init)
    return ps[...,:2]

def trace_all():
    ray_init = lens.sample_ray_2D(R, wavelength, M=M)
    ps, oss = lens.trace_to_sensor_r(ray_init)
    return ps[...,:2], oss

def compute_Jacobian(ps):
    Js = []
    for i in range(1):
        J = torch.zeros(torch.numel(ps))
        for j in range(torch.numel(ps)):
            mask = torch.zeros(torch.numel(ps))
            mask[j] = 1
            ps.backward(mask.reshape(ps.shape), retain_graph=True)
            J[j] = lens.surfaces[i].c.grad.item()
            lens.surfaces[i].c.grad.data.zero_()
        J = J.reshape(ps.shape)

    # get data to numpy
    Js.append(J.cpu().detach().numpy())
    return Js


N = 1
cs_min = 0.010
cs_max = 0.020
cs = np.linspace(cs_min, cs_max, N)
Iss = []
Jss = []
for index, c in enumerate(cs):
    index_string = str(index).zfill(3)
    # load optics
    lens.surfaces[0].c = torch.Tensor(np.array(c))
    lens.surfaces[0].c.requires_grad = True
    
    # show trace figure
    ps, oss = trace_all()
    ax, fig = lens.plot_raytraces(oss, color='b-', show=True)
    ax.axis('off')
    ax.set_title("")
    fig.savefig(save_dir + "layout_trace_" + index_string + ".png", bbox_inches='tight')

    #print(ps - torch.mean(ps, dim=0)[None, ...])
    #print(ps.shape)

    # show spot diagram
    RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(torch.square(ps), axis=-1)))
    ps = render()
    rms_org = RMS(ps)
    print(f'RMS: {rms_org}')
    lens.spot_diagram(ps, xlims=[-15, 15], ylims=[-15, 15], savepath=save_dir + "spotdiagram_" + index_string + ".png", show=False)
    print(ps)
    #print(ps - torch.mean(ps, dim=0)[None, ...])
    print(ps.shape)
    # compute Jacobian
    """ Js = compute_Jacobian(ps)[0]
    print(Js.max())
    print(Js.min())
    ps_ = ps.cpu().detach().numpy()
    fig = plt.figure()
    x, y = ps_[:,0], ps_[:,1]
    plt.plot(x, y, 'b.', zorder=0)
    plt.quiver(x, y, Js[:,0], Js[:,1], color='b', zorder=1)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    fig.savefig(save_dir + "flow_" + index_string + ".png", bbox_inches='tight') """

    # compute images
"""     ray = lens.sample_ray(wavelength.item(), view=0.0, M=2049, sampling='grid')

    I = lens.render(ray)
    I = I.cpu().detach().numpy()
    lm = do.LM(lens, ['surfaces[0].c'], 1e-2, option='diag')
    JI = lm.jacobian(lambda: lens.render(ray)).squeeze()
    J = JI.abs().cpu().detach().numpy()

    Iss.append(I)
    Jss.append(J)
    plt.close()

Iss = np.array(Iss)
Jss = np.array(Jss)
for i in range(N):
    plt.imsave(save_dir + "I_" + str(i).zfill(3) + ".png", Iss[i], cmap='gray')
    plt.imsave(save_dir + "J_" + str(i).zfill(3) + ".png", Jss[i], cmap='gray') """

names = [
    'spotdiagram',
    'layout_trace',
    'I',
    'J',
    'flow'
]

def test_time(wavelengths):
    # load optics
    #lens.surfaces[0].c = torch.Tensor(np.array(0.010))
    #lens.surfaces[0].c.requires_grad = True
    
    # show spot diagram
    for wavelength_ in wavelengths:
        RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(torch.square(ps), axis=-1)))
        ray_init = lens.sample_ray(wavelength_, M=M, R=R, sampling='grid')
        ps = lens.trace_to_sensor(ray_init)[...,:2]
        #rms_org = RMS(ps)
        #print(f'RMS: {rms_org}')
        #lens.spot_diagram(ps, xlims=[-10, 10], ylims=[-10, 10], savepath=save_dir + "spotdiagram_" + index_string + ".png", show=False)
time_start = time.time()
test_time(torch.Tensor(np.linspace(450,650,55)).to(device))
print(f"Elapsed time: {time.time()-time_start}s")