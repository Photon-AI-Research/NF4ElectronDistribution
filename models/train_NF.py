import sys
sys.path.append('./modules')

import torch
import os

import data_preprocessing
import visualization
import dataset
import loader
import model_freia

path = "../simulations/data/data_npy/training_data_npy"
positions = ['lpa', 'Img1', 'Img2', 'Img4', 'UndEnt', 'UndMid', 'Img5', 'Img6']
num_points = 5
num_files = len(positions)
time_stamp = 8
num_inputs = 4
normalize = True
a, b = 0.1, 0.9
dim_phase_space=6

batch_size = 2

num_linear_layers = 3
num_coupling_layers = 7
hidden_size = 128

learning_rate = 1e-5
enable_wandb = True

files = data_preprocessing.prepare_paths(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_tr = dataset.PCDataset(files, 
                 positions=positions,
                 num_points=num_points,
                 num_files=num_files,
                 time_stamp=time_stamp,
                 num_inputs=num_inputs,
                 normalize=normalize, a=a, b=b)

path = "../simulations/data/data_npy/validation_data_npy"
files = data_preprocessing.prepare_paths(path)
        
dataset_val = dataset.PCDataset(files, 
                 positions=positions,
                 num_points=num_points,
                 num_files=num_files,
                 time_stamp=time_stamp,
                 num_inputs=num_inputs,
                 normalize=normalize, a=a, b=b)

print('mins: ', dataset_tr.vmin)
print('maxs: ', dataset_tr.vmax)

print('Create model...')

if enable_wandb:
    visualization.login_wandb(num_linear_layers=num_linear_layers,
                               num_coupling_layers=num_coupling_layers,
                               hidden_size=hidden_size,
                               learning_rate=learning_rate)

model_f = model_freia.PC_NF(dim_condition=num_inputs+len(positions),
                 dim_input=dim_phase_space,
                 num_coupling_layers=num_coupling_layers,
                 num_linear_layers=num_linear_layers,
                 hidden_size=hidden_size,
                 device='cuda',
                 enable_wandb=enable_wandb)

optimizer = torch.optim.Adam(model_f.trainable_parameters, lr=learning_rate, betas=(0.8, 0.9),
                             eps=1e-6, weight_decay=2e-5)
test_pointcloud = '../simulations/data/data_npy/training_data_npy/Track_1.017e-08_2.008e-08_1.778e-03_2.556e-03/Track_1.017e-08_2.008e-08_1.778e-03_2.556e-03-UndMid.npy'

model_f.train_(dataset_tr,
               dataset_val,
               optimizer,
               epochs=1,
               batch_size=2,
               test_epoch=25,
               test_pointcloud=test_pointcloud, log_plots=visualization.plot_3D,
               path_to_models='./RESmodels_freia/')