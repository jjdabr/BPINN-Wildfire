"""
Code to reproduce figure 6 in [1]

[1] Joel Janek Dabrowski, Daniel Edward Pagendam, James Hilton, Conrad Sanderson, 
    Daniel MacKinlay, Carolyn Huston, Andrew Bolt, Petra Kuhnert, "Bayesian 
    Physics Informed Neural Networks for Data Assimilation and Spatio-Temporal 
    Modelling of Wildfires", Spatial Statistics, Volume 55, June 2023, 100746
    https://www.sciencedirect.com/science/article/pii/S2211675323000210
"""

__author__      = "Joel Dabrowski"
__copyright__   = "Copyright Info"
__license__     = "License Name and Info"
__version__     = "0.0.0"

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import time

from dataset import DatasetScaler, level_set_function, c_wind_obstruction_complex
from pinn import PINN_Bayesian
    
train_model = False
n_epochs = 16000
learning_rate = 1e-3
predictive_cost = True
save_file = 'model_parameters/bpinn.pt'

# Environment extents and grid
Nt = 48
Nx = 35
Ny = 35
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0
t_min = 0.0
t_max = 0.1
x_mid = 0.5
y_mid = 0.5
t0 = 0.0
offset = 0.15
dt = (t_max - t_min) / (Nt-1)

# ---------------------------------------------------------------------------- #
# Create the PINN

# MLP parameters
layer_dims_list = [6, 64, 64, 1]
num_batches = 1

# Bayesian MLP prior paramters
prior_sigmas = [0.01, 0.5]
sigma2_i = 1 / (2 * np.pi * 1000)
sigma2_f = 1 / (2 * np.pi * 1.0)
sigma2_a = 1 / (2 * np.pi * 1000)
sigma2_p = 1 / (2 * np.pi * 50)
sigma2_s = 1 / (2 * np.pi * 10.)

model = PINN_Bayesian(layer_dims_list=layer_dims_list, activation_function='tanh', save_file=save_file)

# ---------------------------------------------------------------------------- #
# Dataset scalers
x_norm = DatasetScaler(x_min, x_max)
y_norm = DatasetScaler(y_min, y_max)
t_norm = DatasetScaler(t_min, t_max)
dt = t_norm(dt)

# Generate datasets/sample points
with torch.no_grad():
    # Initial condition
    x0 = torch.linspace(x_min, x_max, Nx)
    y0 = torch.linspace(y_min, y_max, Ny)
    y0 = torch.linspace(y_min, y_max, Ny)
    t0 = torch.linspace(t_min, t_min, 1)
    x0, y0, t0 = torch.meshgrid(x0, y0, t0)
    # Scale
    x0 = x_norm(x0)
    y0 = y_norm(y0)
    t0 = t_norm(t0)
    u0 = level_set_function(x0[:,0,0], y0[0,:,0], x_mid, y_mid, offset)
    s0, wx0, wy0 = c_wind_obstruction_complex(t0[0,0,:], x0[:,0,0], y0[0,:,0])
    # Reshape
    x0 = x0.reshape(-1,1)
    y0 = y0.reshape(-1,1)
    t0 = t0.reshape(-1,1)
    s0 = s0.reshape(-1,1)
    wx0 = wx0.reshape(-1,1)
    wy0 = wy0.reshape(-1,1)
    u0 = u0.reshape(-1,1)
    
    # Collocation data
    x_colloc = torch.linspace(x_min, x_max, Nx)
    y_colloc = torch.linspace(y_min, y_max, Ny)
    t_colloc = torch.linspace(t_min, t_max, Nt)
    x_colloc, y_colloc, t_colloc = torch.meshgrid(x_colloc, y_colloc, t_colloc)
    # Scale
    x_colloc = x_norm(x_colloc)
    y_colloc = y_norm(y_colloc)
    t_colloc = t_norm(t_colloc)
    s_colloc, wx_colloc, wy_colloc = c_wind_obstruction_complex(t_colloc[0,0,:], x_colloc[:,0,0], y_colloc[0,:,0])
    # Reshape
    x_colloc = x_colloc.reshape(-1,1)
    y_colloc = y_colloc.reshape(-1,1)
    t_colloc = t_colloc.reshape(-1,1)
    s_colloc = s_colloc.reshape(-1,1)
    wx_colloc = wx_colloc.reshape(-1,1)
    wy_colloc = wy_colloc.reshape(-1,1)
    
    # Test data
    x_test = torch.linspace(x_min, x_max, Nx)
    y_test = torch.linspace(y_min, y_max, Ny)
    t_test = torch.linspace(t_min, t_max, Nt)
    x_test, y_test, t_test = torch.meshgrid(x_test, y_test, t_test)
    # Scale
    x_test = x_norm(x_test)
    y_test = y_norm(y_test)
    t_test = t_norm(t_test)
    s_test, wx_test, wy_test = c_wind_obstruction_complex(t_test[0,0,:], x_test[:,0,0], y_test[0,:,0])
    # Reshape
    x_test = x_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    t_test = t_test.reshape(-1,1)
    s_test = s_test.reshape(-1,1)
    wx_test = wx_test.reshape(-1,1)
    wy_test = wy_test.reshape(-1,1)
    
t0 = t0.requires_grad_(True)
x0 = x0.requires_grad_(True)
y0 = y0.requires_grad_(True)
x_colloc = x_colloc.requires_grad_(True)
y_colloc = y_colloc.requires_grad_(True)
t_colloc = t_colloc.requires_grad_(True)
t_test = t_test.requires_grad_(True)
x_test = x_test.requires_grad_(True)
y_test = y_test.requires_grad_(True)

# ---------------------------------------------------------------------------- #
# Train the model

if train_model:
    # Model optimiser
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=0.0)
    
    loss_list = []
    model.train()
    for epoch in range(n_epochs):
        tic = time.time()
        optimizer.zero_grad()
        
        # Initial conditions loss
        u0_hat, _, _, _ = model(t0, x0, y0, s0, wx0, wy0)
        # Initial conditions log likelihood
        loglik_i = model.gaussian_log_likelihood(u0_hat, u0.to(model.device), sigma2_i)
            
        # Collocation Points
        u, dudt, dudx, dudy = model(t_colloc, x_colloc, y_colloc, s_colloc, wx_colloc, wy_colloc)
        # Calculate c using the norm vector
        u_norm = torch.sqrt(dudx**2 + dudy**2)
        n_hat_x = dudx / u_norm
        n_hat_y = dudy / u_norm
        c = torch.maximum(s_colloc.to(model.device) + wx_colloc.to(model.device) * n_hat_x + wy_colloc.to(model.device) * n_hat_y, torch.zeros_like(u_norm))
        
        # Physics log likelihood
        phy_loss = torch.sum((dudt + c * u_norm)**2)
        loglik_f = model.gaussian_log_likelihood(dudt + c * u_norm, torch.zeros_like(dudt, device=model.device), sigma2_f)
            
        if predictive_cost:
            # This assumes that the data is sequential.
            # Use torch.no_grad here as u_pred is treated like ground-truth data rather than a calculation
            with torch.no_grad():
                u_pred = u - dt * c * u_norm
            loglik_p = model.gaussian_log_likelihood(u.reshape(Nx, Ny, Nt)[:,:,1:], u_pred.reshape(Nx, Ny, Nt)[:,:,:-1], sigma2_p)
        
        
        # Total log likelihood
        log_likelihood = loglik_i + loglik_f
        if predictive_cost:
            log_likelihood += loglik_p
            
        # Prior over weights
        log_prior = model.log_prior()
        # log of the variational posterior
        log_var_post = model.log_variational_posterior()
        
        # Negative evidence lower bound
        elbo = (log_var_post - log_prior) / num_batches - log_likelihood

        elbo.backward()
        optimizer.step()
        loss_list.append(elbo.item())
        
        # Store latest model parameters
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model.save_file)
        
        # Print progress
        if epoch % 100 == 0:
            toc = time.time() - tic
            print_string1 = 'Epoch {},\tloglik_i = {:.3e}, loglik_f = {:.3e}'.format(epoch, loglik_i, loglik_f)
            print_string2 = ', log_prior = {:.3e}, log_var_post = {:.3e}, elbo = {:.3e}, time-left = {:.1f}min ({:.1f}sec)'.format(log_prior, log_var_post, elbo, (n_epochs-epoch)*toc/60, (n_epochs-epoch)*toc)
            if predictive_cost:
                print_string1 += ', loglik_p = {:.3e}'.format(loglik_p)
            print_string = print_string1+print_string2
            print(print_string)

    # Plot the loss over the epochs
    plt.figure(num=0)
    plt.plot(loss_list)
    plt.title('loss')
    plt.tight_layout()

# ---------------------------------------------------------------------------------------------------------------------
# Plot results

checkpoint = torch.load(model.save_file, map_location=model.device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Evaluate the model over the set of Monte Carlo samples
n_mc_samples = 10
u = np.zeros((n_mc_samples, Nx, Ny, Nt))
dudt = np.zeros((n_mc_samples, Nx, Ny, Nt))
dudx = np.zeros((n_mc_samples, Nx, Ny, Nt))
dudy = np.zeros((n_mc_samples, Nx, Ny, Nt))
for i in range(n_mc_samples):
    u_, dudt_, dudx_, dudy_ = model(t_colloc, x_colloc, y_colloc, s_colloc, wx_colloc, wy_colloc)
    u[i,:,:,:] = u_.reshape((Nx, Ny, Nt)).detach().cpu().numpy()
    dudt[i,:,:,:] = dudt_.reshape((Nx, Ny, Nt)).detach().cpu().numpy()
    dudx[i,:,:,:] = dudx_.reshape((Nx, Ny, Nt)).detach().cpu().numpy()
    dudy[i,:,:,:] = dudy_.reshape((Nx, Ny, Nt)).detach().cpu().numpy()
u_mean = np.mean(u, axis=0)
dudt_mean = np.mean(dudt, axis=0)
dudx_mean = np.mean(dudx, axis=0)
dudy_mean = np.mean(dudy, axis=0)
u_std = np.std(u, axis=0)
dudt_std = np.std(dudt, axis=0)
dudx_std = np.std(dudx, axis=0)
dudy_std = np.std(dudy, axis=0)

x_test = x_test.reshape((Nx, Ny, Nt)).cpu().detach()
y_test = y_test.reshape((Nx, Ny, Nt)).cpu().detach()
wx_test = wx_test.reshape((Nx, Ny, Nt)).cpu()
wy_test = wy_test.reshape((Nx, Ny, Nt)).cpu()
s_test = s_test.reshape((Nx, Ny, Nt)).cpu()

# Plots 
fig1, ax1 = plt.subplots(4,3, figsize=(10,10), num=1, subplot_kw={"projection": "3d"})
fig2, ax2 = plt.subplots(4,3, figsize=(10,10), num=2, subplot_kw={"projection": "3d"})
fig3, ax3 = plt.subplots(4,3, figsize=(10,10), num=3, subplot_kw={"projection": "3d"})
fig4, ax4 = plt.subplots(4,3, figsize=(10,10), num=4, subplot_kw={"projection": "3d"})
fig5, ax5 = plt.subplots(4,3, figsize=(10,10), num=5, sharex=True, sharey=True)
i = 0
j = 0    
# for t in range(min(12, Nt)):
for t in range(0, Nt, Nt//11):
    # Plot u
    ax1[i,j].plot_surface(x_test[:,:,t], y_test[:,:,t], u_mean[:,:,t])
    cset = ax1[i,j].contour(x_test[:,:,t], y_test[:,:,t], u_mean[:,:,t], zdir='z', offset=-0.1)
    ax1[i,j].set_title('t={}'.format(t))
    ax1[i,j].view_init(elev=40, azim=-70)#, roll=0)
    ax2[i,j].grid(True)
    fig1.suptitle('Predicted $u$')

    # Plot du/dx
    ax2[i,j].plot_surface(x_test[:,:,t], y_test[:,:,t], dudx_mean[:,:,t])
    ax2[i,j].set_title('t={}'.format(t))
    ax2[i,j].grid(True)
    fig2.suptitle('Predicted $\partial u / \partial x$')
    
    # Plot du/dy
    ax3[i,j].plot_surface(x_test[:,:,t], y_test[:,:,t], dudy_mean[:,:,t])
    ax3[i,j].set_title('t={}'.format(t))
    ax3[i,j].grid(True)
    fig3.suptitle('Predicted $\partial u / \partial y$')
    
    # Plot du/dt
    ax4[i,j].plot_surface(x_test[:,:,t], y_test[:,:,t], dudt_mean[:,:,t])
    ax4[i,j].set_title('t={}'.format(t))
    ax4[i,j].grid(True)
    fig4.suptitle('Predicted $\partial u / \partial t$')
    
    # Plot the zero-level sets with the obstructions
    for k in range(n_mc_samples):
        ax5[i,j].contour(x_test[:,:,t], y_test[:,:,t], u[k,:,:,t], 0, colors='grey', alpha=0.5)
        
    ax5[i,j].contour(x_test[:,:,t], y_test[:,:,t], u_mean[:,:,t], 0, colors='r')
    grd = 3
    ax5[i,j].quiver(x_test[::grd,::grd,t], y_test[::grd,::grd,t], wx_test[::grd,::grd,t], wy_test[::grd,::grd,t])
    # Plot the obstructions
    ax5[i,j].add_patch(Rectangle((0.0, 0.2), 0.3, 0.6, facecolor="dodgerblue", alpha=0.3, zorder=2))
    ax5[i,j].add_patch(Rectangle((0.7, 0.4), 0.1, 0.1, facecolor="dodgerblue", alpha=0.3, zorder=2))
    ax5[i,j].add_patch(Rectangle((0.7, 0.6), 0.1, 0.1, facecolor="dodgerblue", alpha=0.3, zorder=2))
    # Plot the ground
    ax5[i,j].add_patch(Rectangle((0.0, 0.0), 0.5, 1.0, facecolor='#4fc94f', alpha=0.35, zorder=0))
    ax5[i,j].add_patch(Rectangle((0.5, 0.0), 0.5, 1.0, facecolor='#c2ffc2', alpha=0.35, zorder=0))
    
    ax5[i,j].set_aspect('equal', 'box')
    ax5[i,j].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax5[i,j].grid(True, alpha=0.5)
    ax5[i,j].set_title('t={}'.format(t))
    
    j += 1
    if j >= 3:
        i += 1
        j = 0
    
plt.figure(1)
plt.tight_layout(h_pad=2.0)

plt.figure(2)
plt.tight_layout(h_pad=2.0)
    
plt.figure(3)
plt.tight_layout(h_pad=2.0)
    
plt.figure(4)
plt.tight_layout(h_pad=2.0)

plt.figure(5)
plt.tight_layout(h_pad=0.1)
plt.savefig('readme_images/results.png', dpi=300)
plt.show()
