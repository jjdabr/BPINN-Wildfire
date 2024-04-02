"""
Bayesian Physics Informed Neural Network (B-PINN)

[1] Joel Janek Dabrowski, Daniel Edward Pagendam, James Hilton, Conrad Sanderson, 
    Daniel MacKinlay, Carolyn Huston, Andrew Bolt, Petra Kuhnert, "Bayesian 
    Physics Informed Neural Networks for Data Assimilation and Spatio-Temporal 
    Modelling of Wildfires", Spatial Statistics, Volume 55, June 2023, 100746
    https://www.sciencedirect.com/science/article/pii/S2211675323000210
"""

__author__      = "Joel Janek Dabrowski"
__license__     = "MIT license"
__version__     = "0.0.0"


import torch
import torch.nn as nn
from tqdm import tqdm
import math
from mlp import MLP_Bayesian

class PINN_Bayesian(nn.Module):
    def __init__(self, layer_dims_list, activation_function='relu', save_file='model_parameters/bpinn.pt'):
        """
        Constructor

        :param layer_dims_list: list containing the number of units in each 
            layer (including the input and output layers) of the MLP for u_model
        :param activation_function: activation function in the MLP. Options: 
            relu, tanh, and silu. Defaults to 'relu'
        :param save_file: path and name of the file in which to store the model 
            parameters, defaults to 'model_parameters/bpinn.pt'
        """
        super(PINN_Bayesian, self).__init__()
        self.layer_dims_list = layer_dims_list
        # Number of Monte Carlo samples of the weights z drawn from the 
        # variational posterior
        self.n_mc_samples = 1
        # Model
        self.u_model = MLP_Bayesian(layer_dims_list, activation_function=activation_function)
        self.save_file = save_file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.u_model.to(self.device)

    def log_prior(self):
        """
        Calculate the log of the prior distribution of the weights over all 
        layers in the network
        
        :return: log prior probability (scalar)
        """
        log_prior = 0
        for layer in self.u_model.layers:
            log_prior += layer.log_prior
        return log_prior

    def log_variational_posterior(self):
        """
        Calculate the log of the posterior of the weights given the data over 
        all layers in the network
        
        :return: log of the posterior probability (scalar)
        """
        log_variational_posterior = 0
        for layer in self.u_model.layers:
            log_variational_posterior += layer.log_variational_posterior
        return log_variational_posterior


    def gaussian_log_likelihood(self, y, y_hat, sigma2):
        """
        Calculate the Gaussian log likelihood of the data given the model.
        The Gaussian has mean at y_hat and standard deviation given by 
        noise_tol. The log likelihood is calculated for the data sample y.
        
        :param y: The ground truth data
        :param y_hat: The model prediction
        :param noise_tol: the standard deviation of the distribution
        :return: the log likelihood of the data given the model (scalar)
        """
        log_gauss = (-math.log(math.sqrt(2 * math.pi * sigma2))
                     - ((y - y_hat) ** 2) / (2 * sigma2)).sum()
        return log_gauss

    
    def forward(self, t, x, y, s, wx, wy):
        """
        Predict the level-set function given a set of inputs

        :param t: time tensor with shape [batch_size, 1]
        :param x: tensor over x-spatial dimension with shape [batch_size, 1]
        :param y: tensor over y-spatial dimension with shape [batch_size, 1]
        :param s: fire-front speed constant with shape [batch_size, 1]
        :param wx: wind speed in the x-direction with shape [batch_size, 1]
        :param wy: wind speed in the x-direction with shape [batch_size, 1]
        :return: the level set function prediction and the partial derivatives 
            of the level set function with respect to time and space
        """
        t = t.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        s = s.to(self.device)
        wx = wx.to(self.device)
        wy = wy.to(self.device)
        # Compute u_t using the neural network
        model_in = torch.cat((t, x, y, s, wx, wy), dim=1).to(self.device)
        u = self.u_model(model_in, sample=True)
        # Compute du/dt
        dudt = torch.autograd.grad(outputs=u, inputs=t, grad_outputs=torch.ones_like(u, device=self.device), create_graph=True)[0]
        # Compute du/dx
        dudx = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u, device=self.device), create_graph=True)[0]
        # Compute du/dy
        dudy = torch.autograd.grad(outputs=u, inputs=y, grad_outputs=torch.ones_like(u, device=self.device), create_graph=True)[0]
        return u, dudt, dudx, dudy
    