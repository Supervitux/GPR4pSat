import torch
import gpytorch
import random
import numpy as np
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from scipy.sparse import load_npz, save_npz
import os
import json
from sklearn.model_selection import train_test_split
import math
import scipy.sparse

import time # by Vitus

# exactGP model with constant mean and scaled RBF kernel
class ExactGPModel(gpytorch.models.ExactGP): 
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() 
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_constraint = gpytorch.constraints.Interval(lower_bound = 1e-6, upper_bound = 1e6)),outputscale_constraint=gpytorch.constraints.Interval(lower_bound = 1e-6, upper_bound = 1e6) ) 

        #print("Constraint? \n", self.covar_module.base_kernel.lengthscaleconstraints)
        #self.covar_module.base_kernel.lengthscale._constraints(LowerThan(1e6))
        #print("Constant initialized: \n", self.mean_module.__dict__)
        #print("RBF Kernel initialized: \n", self.covar_module.__dict__)
        #base_covar_module = gpytorch.kernels.ScaleKernel(
        #    gpytorch.kernels.RBFKernel())
        #self.covar_module = base_covar_module

        # Addition for the sake of using multiple GPUs (all mentions of "n_devices")
        #self.covar_module = gpytorch.kernels.MultiDeviceKernel(
        #        base_covar_module, device_ids=range(n_devices),
        #) 

    def forward(self, x): 
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPytorchGPModel(): 
    def __init__(self, x_train, y_train, likelihood):
        self.random_seed = 1234 
        self.likelihood = likelihood.double() 
        self.params = dict() 
        
        self.normalize_y = True
        if self.normalize_y:
            self._y_train_mean = np.mean(y_train, axis=0)
            self._y_train_std = np.std(y_train, axis=0)
        else:
            self._y_train_mean = 0
            self._y_train_std = 1

        
        # normalize outputs
        if self.normalize_y:
            y_train = (y_train - self._y_train_mean) / self._y_train_std

        # double precision for numerical stability
        train_y = (torch.from_numpy(y_train)).double() 
        train_x = (torch.from_numpy(x_train)).double()

        self.use_gpu = True
        # for GPU training if available
        if torch.cuda.is_available() and self.use_gpu:
            print("GPUs activated!") 
            self.model = ExactGPModel(train_x, train_y, self.likelihood).cuda()
        # if GPU is not available, CPU used
        else:
            print("CPUs activated!")
            self.model = ExactGPModel(train_x, train_y, self.likelihood)


    def normalize(self, y):
        return (y - self._y_train_mean) / self._y_train_std


    def unnormalize(self, y):
        return (y*self._y_train_std) + self._y_train_mean


    def fit(self, x_train,
              y_train,
              x_valid,
              y_valid,
              torch_optimizer=torch.optim.LBFGS,
              lr=0.1,
              max_epochs=500):
        print("Start fit with optimizer:")
        print(torch_optimizer)
        losses = []
        noises = []
        raw_noise = []
              
        mae_loss = torch.nn.L1Loss() # Maybe no point having this here? It is needed for predicting?
        
        if torch.cuda.is_available() and self.use_gpu:
            x_train = (torch.from_numpy(x_train)).double().cuda()
            y_train = (torch.from_numpy(y_train)).double().cuda()
            x_valid = (torch.from_numpy(x_valid)).double().cuda()
            y_valid = (torch.from_numpy(y_valid)).double().cuda()
        else:
            x_train = (torch.from_numpy(x_train)).double()
            y_train = (torch.from_numpy(y_train)).double()
            x_valid = (torch.from_numpy(x_valid)).double()
            y_valid = (torch.from_numpy(y_valid)).double()
        
        optimizer = torch_optimizer(self.model.parameters(), lr=lr)
        
        # set model and likelihood to double precision
        self.model = self.model.double()
        self.likelihood = self.likelihood.double() 
        
        # set model and likelihood to train mode
        self.model.train() 
        self.likelihood.train()
                
        # exact marginal likelihood for training with exact GP:        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)  
        loss_prev = np.Inf
        
        # normalize y, 
        if self.normalize_y:
            y_train = self.normalize(y_train) 
       
        t_start = time.perf_counter()   # Start fitting counter
        print("Some epochs")
        for i in range(max_epochs):
            # to opt for exact Cholexky instead of CG approximation
            with gpytorch.settings.fast_computations(solves=False,log_prob=False, covar_root_decomposition=True): 
                # this is done to compute loss value, no optimization step taken here
                optimizer.zero_grad()
                output = self.model(x_train)
                # compute current loss value and noise values
                loss = -mll(output, y_train) 
                losses.append(loss.item())
                noises.append(self.model.likelihood.noise.item())
                raw_noise.append(self.model.likelihood.raw_noise.item())


                # function of single optimization step form optimizer
                # Some optimizers such as LBFGS require this
                def closure():
                    optimizer.zero_grad()
                    output = self.model(x_train)
                    loss = -mll(output, y_train)
                    loss.backward()
                    return loss

                # take one full optimization step
                optimizer.step(closure)  

            
            tol = 10**(-5) # stopping tolerance, default: 10**-4

            # stopping condition evaluated every ith itertaion
            if i % 1 == 0: 
                if loss_prev - loss <= tol:
                    print("Early Stopping by tol")
                    break
                    self.model.train()
                    self.likelihood.train()
                        

                loss_prev = loss # update loss
                
            print(
                f"Iter {i+1}/{max_epochs} - MLL {loss.item()} - const {self.model.covar_module.outputscale.item()} -length {self.model.covar_module.base_kernel.lengthscale.item()} - noise {self.model.likelihood.noise.item()}"
            )

        t_end = time.perf_counter()
        time_fit = t_end - t_start


        return losses, noises, raw_noise, time_fit
    
    def predict(self, x_valid):
        print("Predicting ...")
        # setting the model to eval mode
        self.model.eval() 
        self.likelihood.eval()

        if torch.cuda.is_available() and self.use_gpu:
            x_valid = (torch.from_numpy(x_valid)).double().cuda()
        else:
            x_valid = (torch.from_numpy(x_valid)).double()

        # Cholescy instead of CG
        with torch.no_grad(), gpytorch.settings.fast_computations(solves=False, log_prob=False, covar_root_decomposition=True):
            t_start = time.perf_counter()
            pred = self.likelihood(self.model(x_valid.double()))
            t_end = time.perf_counter()
            time_pred = t_end - t_start
        return pred, time_pred
   
    def mae_loss(self, y_real, y_pred):
        mae_loss1 = torch.nn.L1Loss()    
        if torch.cuda.is_available() and self.use_gpu:
            y_real = (torch.from_numpy(y_real)).double().cuda()
            y_pred = (y_pred.mean).double().cuda()
        else:
            y_real = (torch.from_numpy(y_real)).double()
            y_pred = (y_pred.mean).double()
        
        if self.normalize_y:
            val_ = mae_loss1(self.unnormalize(y_pred), y_real)
        else:
            val_ = mae_loss1(y_pred, y_real)
        return val_.item()

    def get_params(self):
        self.params = {
            "constant_value" : self.model.covar_module.outputscale.item(),
            "length_scale"   : self.model.covar_module.base_kernel.lengthscale.item()
        }
        return self.params

    
    # set paremeters for gamma priors for lenghtscale and constant scale
    def set_params(self,consta, lenght_scalea, constb, lenght_scaleb):
        outputscale_prior = gpytorch.priors.GammaPrior(consta,constb )
        lengthscale_prior = gpytorch.priors.GammaPrior(lenght_scalea, lenght_scaleb)
        if torch.cuda.is_available() and self.use_gpu:
            self.model.covar_module.outputscale_prior  = outputscale_prior.cuda()
            self.model.covar_module.base_kernel.lengthscale_prior = lengthscale_prior.cuda()
        else:
            self.model.covar_module.outputscale_prior  = outputscale_prior
            self.model.covar_module.base_kernel.lengthscale_prior = lengthscale_prior        
        init_const = self.model.covar_module.outputscale_prior.sample((1,))
        init_length = self.model.covar_module.base_kernel.lengthscale_prior.sample((1,))
        self.model.covar_module.outputscale = init_const
        self.model.covar_module.base_kernel.lengthscale = init_length
        # If above should be tensor try: torch.tensor([60.0])        

    def set_params_nopriors(self,init_const, init_length, init_noise):
        self.model.covar_module.outputscale = init_const
        self.model.covar_module.base_kernel.lengthscale = init_length
        self.model.likelihood.noise = init_noise


