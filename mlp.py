"""
Bayesian Multi-Layered Perceptron (MLP)

[1] Joel Janek Dabrowski, Daniel Edward Pagendam, James Hilton, Conrad Sanderson, 
    Daniel MacKinlay, Carolyn Huston, Andrew Bolt, Petra Kuhnert, "Bayesian 
    Physics Informed Neural Networks for Data Assimilation and Spatio-Temporal 
    Modelling of Wildfires", Spatial Statistics, Volume 55, June 2023, 100746
    https://www.sciencedirect.com/science/article/pii/S2211675323000210
[2] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra, 
    "Weight Uncertainty in Neural Networks", 2015
    https://arxiv.org/abs/1505.05424
"""

__author__      = "Joel Dabrowski"
__copyright__   = "Copyright Info"
__license__     = "License Name and Info"
__version__     = "0.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Gaussian(object):
    """
    Re-parametrised Gaussian distribution where
        z = mu + log(1 + exp(rho)) * N(0, I)
    """
    def __init__(self, mu, rho):
        """
        Constructor

        :param mu: mean with shape [batch_size, dim]
        :param rho: re-parametrised standard deviation parameter with shape 
            [batch_size, dim]
        """
        super().__init__()
        # Mean of Gaussian
        self.mu = mu
        # variance parameter where sigma = log(1 + exp(rho))
        self.rho = rho
        # epsilon = N(0, I)
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        """
        Return the standard deviation, which is computed from self.rho
        """
        # sigma = log(1 + exp(rho))
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        """
        Sample from the Gaussian distribution

        :return: Sample with shape [batch_size, dim] 
        """
        # Sample epsilon from N(0,I)
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        # Return z = mu + log(1 + exp(rho)) * N(0, I)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        """
        Compute the log probability of the factorized Gaussian:
          p(z) = \prod_j N(z^{(j)}| \mu, \sigma^2)
               = \sum_j [-log(2 pi) - log(sigma) - (x - \mu)^2 / (2 \sigma^2)]
        :param input: input tensor with shape [batch_size, dim] 
        :return: scalar log probability value
        """
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    """
    This implements the spike and slap prior with Gaussian distributions denoted 
    by N(mu, sigma) (see [2]):
        p(z) ~ pi N(0, sigma_1^2) + (1-pi) N(0, sigma_2^2)
    where
        sigma_1 > sigma_2 and sigma_2 << 1
    """
    def __init__(self, pi, sigma1, sigma2):
        """
        Constructor

        :param pi: mixture weight (scalar)
        :param sigma1: standard deviation of the slab mixture as a tensor with 
            shape [1]
        :param sigma2: standard deviation of the spike mixture as a tensor with 
            shape [1]
        """
        super().__init__()
        # Weighting between the two Gaussians
        self.pi = pi
        # Standard deviation of the first and second Gaussians
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        # Distributions of the first and second Gaussians
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        """
        Compute the log probability of the scale mixture distribution, which is 
        the sum ofthe log probability of each Gaussian (see the Gaussian class).

        :param input: input tensor with shape [batch_size, dim] 
        :return: scalar log probability value
        """
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    """
    Custom linear layer which replaces nn.Linear. The layer contains mean and 
    rho parameters for the model weights, the model biases, and the spike and
    slab prior for the weights and biases.
    """
    def __init__(self, in_features, out_features):
        """
        Constructor

        :param in_features: size of each input sample
        :param out_features: size of each output sample
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        # Use the reparameterised Gaussian
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        # Use the reparameterised Gaussian
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        PI = 0.5
        SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
        SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=True, calculate_log_probs=False):
        """
        Apply the Bayesian linear layer to an input.

        :param input: input tensor with shape [*, in_features]
        :param sample: if true return a sample from the distribution otherwise 
            return the means, defaults to True. This is typically set to 
        :param calculate_log_probs: if true, compute the log probabilities which 
            can be retrieved as objects of the class, defaults to False
        :return: the output tensor with shape [*, out_features]
        """
        if self.training or sample:
            # When training or sampling, we take samples from the 
            # re-parametrised Gaussians
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # Otherwise, we return the means
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            # Update the prior and variational posterior based on the samples
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            # No samples were taken, so zero the prior and variational posterior
            self.log_prior = 0
            self.log_variational_posterior = 0
        return F.linear(input, weight, bias)


class MLP_Bayesian(nn.Module):
    """
    MLP model with a Bayesian Linear function.
    """
    
    def __init__(self, layer_dims_list, activation_function='relu'):
        """
        Constructor
        
        :param layer_dims_list: A list of the dimensions of each layer including 
            the input and output layers (referred to as input_dim and output_dim 
            respectively)
        :param activation_function: Activation function to use: 'relu' (default) 
            or 'tanh'
        """
        super(MLP_Bayesian, self).__init__()
        self.layer_dims_list = layer_dims_list
        self.n_layers = len(layer_dims_list) - 1
        self.activation_function = activation_function
        assert len(layer_dims_list) > 1, "Dimensions of at least two layers (input and output) need to be specified"
        # Layers
        self.layers = nn.ModuleList([BayesianLinear(layer_dims_list[i-1], layer_dims_list[i])
                                     for i in range(1, len(layer_dims_list))])
    
    def forward(self, x, sample=True):
        """
        Forward propagation of the MLP model.
        
        :param x: Input data in the form [batch_size, input_dim]
        :param sample: if true return a sample from the distribution otherwise 
            return the means, defaults to True
        :return: the output tensor with shape [batch_size, output_dim]
        """
        output = self.layers[0](x, sample)
        for i in range(1, self.n_layers):
            if self.activation_function == 'relu':
                output = torch.relu(output)
            elif self.activation_function == 'tanh':
                output = torch.tanh(output)
            elif self.activation_function == 'silu':
                output = F.silu(output)
            output = self.layers[i](output, sample)
        return output
