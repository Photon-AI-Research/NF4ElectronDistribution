import sys
sys.path.append('./modules')

import nflows
import numpy as np
import random
import os

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.standard import AffineTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet

import torch
from torch import nn
from torch import optim
import wandb
import matplotlib.pyplot as plt

import data_preprocessing
import loader
import horovod.torch as hvd

torch.autograd.set_detect_anomaly(True)
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['lines.linewidth'] = 6
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

class PC_MAF(nn.Module):
    def __init__(self, dim_condition,
                 dim_input,
                 num_coupling_layers=1,
                 hidden_size=128,
                 device='cpu',
                 enable_wandb=False,
                 use_hvd=False):
        
        '''
        Masked autoregressive flows model from https://papers.nips.cc/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html
        Args:
            dim_condition(integer): dimensionality of condition
            dim_input(integer): dimensionality of input
            num_coupling_blocks(integer): number of coupling blocks in the model
            hidden_size(integer): number of hidden units per hidden layer in subnetworks
            device: "cpu" or "cuda"
            enable_wandb(boolean): True to watch training progress at wandb.ai
            use_hvd(boolean): True if to use horovod for distributed training
            
        '''

        super().__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_coupling_layers = num_coupling_layers
        self.hidden_size = hidden_size

        self.dim_input = dim_input
        self.dim_condition = dim_condition
        self.model = self.init_model().to(self.device)
        
        self.num_points_forward = 10000
        self.enable_wandb = enable_wandb
        self.use_hvd = use_hvd
        self.watch_rank = 0
        if self.use_hvd:
            self.watch_rank = hvd.rank()
    
    def init_model(self):
        base_dist = nflows.distributions.normal.StandardNormal(shape=[self.dim_input])
        transforms = []
        for _ in range(self.num_coupling_layers):
            transforms.append(ReversePermutation(features=self.dim_input))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.dim_input, 
                                                                  hidden_features=self.hidden_size, 
                                                                  context_features=self.dim_condition,
                                                                  use_residual_blocks=True))
        transform = CompositeTransform(transforms)

        return Flow(transform, base_dist).to(self.device)

    def forward(self, x, p):
        loss = self.model(x, c=p)
        return loss

    def train_(self, 
               dataset_tr,
               dataset_val,
               optimizer,
               epochs=100,
               batch_size=2,
               test_epoch=25,
               test_pointcloud=None, log_plots=None,
               path_to_models='./RESmodels_freia/'):
        '''
        Train model
        Args:
            dataset_tr(torch dataset, see modules/dataset.py): dataset with all needed information about data to be used for training
            dataset_val: dataset with all needed information about data to be used for validation
            optimizer: torch.optim optimizer 
            epochs(integer): number of training iterations
            batch_size(integer): size of a batch in loader
            test_epoch(integer): at each test_epoch save perform validation, save pre-trained model is path_to_models is given and register logs to wandb
            test_pointcloud(string): path to a test pointcloud to watch recontsructions at wandb.ai
            log_plots(function): function to log necessary plots of groundtruth and reconstruction, should be not None in case test_pointcloud is not None
            path_to_models(string): path where to save pre-trained models if None then models won't be saved
        '''
        
        self.model.train()
        
        if self.use_hvd:
            hvd.broadcast_parameters(self.state_dict(), root_rank=0)

        loader_tr = loader.get_loader(dataset_tr, batch_size=batch_size, use_hvd=self.use_hvd)

        for i_epoch in range(epochs):
            if self.watch_rank == 0:
                loss_avg = []
            
            for batch_features in loader_tr:
                if dataset_tr.normalize:
                    batch_features = data_preprocessing.normalize_point(batch_features, dataset_tr.vmin, dataset_tr.vmax)
                #merge batch dimension with number of particles, 
                #the resulting shape of each batch should be (batch_size*num_particles, num_in_features+dim_input)
                if len(batch_features.shape) == 3:
                    batch_features = batch_features.reshape(batch_features.shape[0]*batch_features.shape[1],
                                                            batch_features.shape[2])
                
                optimizer.zero_grad()
                y = batch_features[:, :self.dim_condition].float().to(self.device)
                x = batch_features[:, self.dim_condition:].float().to(self.device)

                loss = - self.model.log_prob(inputs=x, context=y).mean()
                loss.backward()
                optimizer.step()
                
                if self.watch_rank == 0:
                    loss_avg.append(loss.item())
                if self.enable_wandb and (self.watch_rank == 0):
                    wandb.log({'NLL-loss': float(loss.item())})
                
            if self.enable_wandb and (self.watch_rank == 0):
                wandb.log({'NLL-loss_avg': sum(loss_avg)/len(loss_avg)})

            if i_epoch % test_epoch == 0:
                if ((not test_pointcloud == None) 
                    and (not log_plots == None) 
                    and self.enable_wandb 
                    and (self.watch_rank == 0)):
                    
                    log_plots(test_pointcloud, dataset_tr, self)

                if path_to_models != None:
                    if self.watch_rank == 0:
                        if not os.path.exists(path_to_models):
                            os.makedirs(path_to_models)
                        self.save_checkpoint(self.model, optimizer, path_to_models, i_epoch)
                        print("epoch : {}/{},\n\tloss_avg = {:.15f}"
                              .format(i_epoch + 1,
                                      epochs,
                                      sum(loss_avg)/len(loss_avg)))
                        self.validation(dataset_val, dataset_tr.vmin, dataset_tr.vmax, batch_size)

    def validation(self, dataset_val, vmin, vmax, batch_size):
        loader_val = loader.get_loader(dataset_val, batch_size=batch_size, use_hvd=self.use_hvd)
        with torch.no_grad():
            loss_avg = []
            for batch_features in loader_val:
                if dataset_val.normalize:
                    #normalize for vmin/vmax that were used for training data
                    batch_features = data_preprocessing.normalize_point(batch_features, vmin, vmax)
                #merge batch dimension with number of particles, 
                #the resulting shape of each batch should be (batch_size*num_particles, num_in_features+dim_input)
                if len(batch_features.shape) == 3:
                    batch_features = batch_features.reshape(batch_features.shape[0]*batch_features.shape[1],
                                                                batch_features.shape[2])
                
                y = batch_features[:, :self.dim_condition].float().to(self.device)
                x = batch_features[:, self.dim_condition:].float().to(self.device)

                loss = - self.model.log_prob(inputs=x, context=y).mean()
                if self.watch_rank == 0:
                    loss_avg.append(loss.item())
            if self.enable_wandb and (self.watch_rank == 0):
                wandb.log({'NLL-loss_val': sum(loss_avg)/len(loss_avg)})
            print("Validation:\n\tloss_avg = {:.15f}"
                      .format(sum(loss_avg)/len(loss_avg)))

    def save_checkpoint(self, model, optimizer, path, epoch):
        if self.watch_rank == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, path + 'model_' + str(epoch))
