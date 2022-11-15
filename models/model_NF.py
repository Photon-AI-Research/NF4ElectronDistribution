import sys
sys.path.append('./modules')

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import os
import random

import FrEIA.framework as Ff
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import data_preprocessing
import loader

import wandb

torch.autograd.set_detect_anomaly(True)
#plt.rcParams['image.cmap'] = 'bwr'
#plt.set_cmap('bwr')

class PC_NF(nn.Module):
    def __init__(self, 
                 dim_condition,
                 dim_input,
                 num_coupling_layers=1,
                 num_linear_layers=1,
                 hidden_size=128,
                 device='cpu',
                 enable_wandb=False):
        
        '''
        Conditional normalizing flow model from https://arxiv.org/abs/1907.02392
        Args:
            dim_condition(integer): dimensionality of condition
            dim_input(integer): dimensionality of input
            num_coupling_blocks(integer): number of coupling blocks in the model
            num_linear_layers(integer): number of hidden linear layers in subnetworks
            hidden_size(integer): number of hidden units per hidden layer in subnetworks
            device: "cpu" or "cuda"
            enable_wandb(boolean): True to watch training progress at wandb.ai
            
        '''

        super().__init__()
        self.device = device
        self.num_coupling_layers = num_coupling_layers
        self.num_linear_layers = num_linear_layers
        self.hidden_size = hidden_size

        self.dim_input = dim_input
        self.dim_condition = dim_condition
        self.model = self.init_model().to(self.device)

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        self.grads = []
        
        self.num_points_forward = 100000
        
        self.enable_wandb = enable_wandb
    
    def init_model(self):
        nodes = [InputNode(self.dim_input, name='input')]
        cond = Ff.ConditionNode(self.dim_condition) 
        
        for k in range(0, self.num_coupling_layers):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':self.subnet_fc, 'clamp':2.0},
                              name=F'coupling_{k}', conditions=cond))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed':k},
                              name=F'permute_{k}'))
        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def subnet_fc(self, c_in, c_out):
        layers=[nn.Linear(c_in, self.hidden_size), nn.LeakyReLU()]
        for i in range(self.num_linear_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.LeakyReLU())
            
        layers.append(nn.Linear(self.hidden_size,  c_out))
        mlp = nn.Sequential(*layers)
        for lin_layer in mlp:
            if(isinstance(lin_layer, nn.Linear)):
                nn.init.constant_(lin_layer.bias, 0)
                nn.init.xavier_uniform_(lin_layer.weight)

        nn.init.constant_(mlp[-1].bias, 0)
        nn.init.constant_(mlp[-1].weight, 0)
        return mlp
    
    def forward(self, x, p):
        z, jac = self.model(x, c=p)
        return z, jac

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
        
        #loader_tr(torch data loader, see modules/loader.py): iterator over training data, each returned elem should be 
        loader_tr = loader.get_loader(dataset_tr, batch_size=batch_size)
        self.model.train()
        
        for i_epoch in range(epochs):
            loss_avg = []
            loss_z = []
            loss_j = []
            loss_bl = []

            for batch_features in loader_tr:
                if dataset_tr.normalize:
                    batch_features = data_preprocessing.normalize_point(batch_features, dataset_tr.vmin, dataset_tr.vmax)
                #merge batch dimension with number of particles, 
                #the resulting shape of each batch should be (batch_size*num_particles, num_in_features+dim_input)
                if len(batch_features.shape) == 3:
                    batch_features = batch_features.reshape(batch_features.shape[0]*batch_features.shape[1],
                                                            batch_features.shape[2])
                optimizer.zero_grad()
    
                x = batch_features[:, :self.dim_condition].float().to(self.device)
                y = batch_features[:, self.dim_condition:].float().to(self.device)

                z, log_j = self.forward(y, x)
                loss = 0.
                
                loss_z.append(float(torch.mean(z**2) / 2))
                loss_j.append(float(torch.mean(log_j))/2)
                loss = torch.mean(z**2) / 2 - torch.mean(log_j)/2

                torch.nn.utils.clip_grad_norm_(self.trainable_parameters, 10.)
                loss_avg.append(loss.item())

                loss.backward()
                optimizer.step()

                if self.enable_wandb:
                    wandb.log({'loss_z_tr': loss_z[-1],
                               'loss_jac_tr': loss_j[-1],
                               'loss_tr': loss_avg[-1]})
                
            if self.enable_wandb:
                wandb.log({'loss_z_avg_tr': sum(loss_z)/len(loss_z),
                           'loss_jac_avg_tr': sum(loss_j)/len(loss_j),
                           'loss_avg_tr': sum(loss_avg)/len(loss_avg)})

            if i_epoch % test_epoch == 0: 
                if (not test_pointcloud == None) and (not log_plots == None) and self.enable_wandb:
                    log_plots(test_pointcloud, dataset_tr, self)

                if path_to_models != None:
                    if not os.path.exists(path_to_models):
                        os.makedirs(path_to_models)
                    self.save_checkpoint(self.model, optimizer, path_to_models, i_epoch)

                print("epoch : {}/{},\n\tloss_avg = {:.15f},\n\tloss_z = {:.15f},\n\tloss_j = {:.15f}"
                      .format(i_epoch + 1, epochs, 
                              sum(loss_avg)/len(loss_avg),
                              sum(loss_z)/len(loss_z),
                              sum(loss_j)/len(loss_j)))
                self.validation(dataset_val, dataset_tr.vmin, dataset_tr.vmax, batch_size)
                
    def validation(self, dataset_val, vmin, vmax, batch_size):
        loader_val = loader.get_loader(dataset_val, batch_size=batch_size)
        with torch.no_grad():
            loss_z = []
            loss_j = []
            loss_avg = []
            for batch_features in loader_val:
                #merge batch dimension with number of particles, 
                #the resulting shape of each batch should be (batch_size*num_particles, num_in_features+dim_input)
                if len(batch_features.shape) == 3:
                    batch_features = batch_features.reshape(batch_features.shape[0]*batch_features.shape[1],
                                                                batch_features.shape[2])
                if dataset_val.normalize:
                    #normalize for vmin/vmax that were used for training data
                    batch_features = data_preprocessing.normalize_point(batch_features, vmin, vmax)

                x = batch_features[:, :self.dim_condition].float().to(self.device)
                y = batch_features[:, self.dim_condition:].float().to(self.device)

                z, log_j = self.forward(y, x)
                loss_z.append(float(torch.mean(z**2) / 2))
                loss_j.append(float(torch.mean(log_j))/2)
                loss = torch.mean(z**2) / 2 - torch.mean(log_j)/2
                loss_avg.append(loss.item())
            if self.enable_wandb:
                wandb.log({'loss_z_val': sum(loss_z)/len(loss_z),
                       'loss_jac_val': sum(loss_j)/len(loss_j),
                       'loss_val': sum(loss_avg)/len(loss_avg)})
            print("Validation:\n\tloss_avg = {:.15f},\n\tloss_z = {:.15f},\n\tloss_j = {:.15f}"
                      .format(sum(loss_avg)/len(loss_avg),
                              sum(loss_z)/len(loss_z),
                              sum(loss_j)/len(loss_j)))

    def save_checkpoint(self, model, optimizer, path, epoch):
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(state, path + 'model_' + str(epoch))
