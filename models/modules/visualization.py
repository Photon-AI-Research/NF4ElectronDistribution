import os
import numpy as np
import torch

import wandb

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as tick
from matplotlib import cm

import data_preprocessing
from data_preprocessing import extend_to_hotvec

def model_forward(model, cond):
    with torch.no_grad():
        if (model.__class__.__name__ == "PC_MAF"):
            pc_pr = (model.model.sample(1, cond)).squeeze(1)
        if (model.__class__.__name__ == "PC_NF"):
            z = torch.randn(model.num_points_forward, model.dim_input).to(model.device)
            pc_pr, _ = model.model(z, c=cond, rev=True)
    return pc_pr

def plot_3D(pointcloud, dataset, model):
    pc = np.load(pointcloud)
    pc = extend_to_hotvec(pointcloud, pc, dataset.positions, dataset.num_inputs)
    fig = plt.figure(figsize=(15,7))
    fig.suptitle(('emmit_x: ' + str(pc[0,0]) + '; ' +
                  'emmit_y: ' + str(pc[0,1]) + '; ' +
                  'beta_x: ' + str(pc[0,2]) + '; ' +
                  'beta_y: ' + str(pc[0,3])), fontsize=16)
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    norm = matplotlib.colors.Normalize(vmin=np.min(pc[:, -1]), vmax=np.max(pc[:, -1]))

    num_in_features = dataset.num_inputs + len(dataset.positions)
    ax.scatter(pc[:, 4 + num_in_features],
               pc[:, 2 + num_in_features],
               pc[:, num_in_features], c=plt.cm.jet(norm(pc[:, -1])))
    
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    plt.colorbar(m, shrink=0.6, pad=0.1)

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_ylabel('x')
    
    ax.set_title('GT')
    
    pc_pr = model_forward(model, data_preprocessing.normalize_point(
                                           torch.from_numpy(pc[:model.num_points_forward, :model.dim_condition]).float().to(model.device),
                                           dataset.vmin[:model.dim_condition].to(model.device),
                                           dataset.vmax[:model.dim_condition].to(model.device),
                                           dataset.a, dataset.b))

    if dataset.normalize:
        pc_pr = data_preprocessing.denormalize_point(pc_pr,
                                      dataset.vmin[model.dim_condition:].to(model.device),
                                      dataset.vmax[model.dim_condition:].to(model.device),
                                      dataset.a, dataset.b)
    pc_ = pc_pr.detach().cpu().numpy()
                    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    #take spacial coordinates for scatter coordinates and energy value for color
    ax.scatter(pc_[:, 4], pc_[:, 2], pc_[:, 0], c=plt.cm.jet(norm(pc_[:, -1])))

    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    plt.colorbar(m, shrink=0.6)

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_ylabel('x')

    ax.set_title('Rec')

    image = wandb.Image(fig)
    wandb.log({"PC": image})
    plt.close()
    
    pc[:, 8] = pc[:, 8]*3e8 - np.mean(pc[:, 8]*3e8)
    fig_, ax_ = plot_slicing(pointcloud, pc, pc_, num_slices=100)
    image = wandb.Image(fig_)
    wandb.log({"SliceEmmit": image})
    
def plot_slicing(fname, pc, pc_pr, num_slices=100,
                figsize1=27, figsize2=21):
    fig, axs = plt.subplots(3, 3, figsize=(figsize1,figsize2))
    fig.suptitle(('emmit_x: ' + str(pc[0,0]) + '; ' +
              'emmit_y: ' + str(pc[0,1]) + '; ' +
              'beta_x: ' + str(pc[0,2]) + '; ' +
              'beta_y: ' + str(pc[0,3])), fontsize=16)

    slices = [np.min(pc[:, 8]) + (np.max(pc[:, 8]) - np.min(pc[:, 8])) * i/num_slices for i in range(num_slices)]
    pc_ = np.concatenate((pc, np.zeros((pc.shape[0], 1))), axis=1)
    pc_pr_ = np.concatenate((pc_pr, np.zeros((pc_pr.shape[0], 1))), axis=1)

    for ind in range(len(slices)-1):
        pc_[:, -1][(pc_[:, 8]>=slices[ind]) & (pc_[:, 8]<=slices[ind+1])] = ind
        pc_pr_[:, -1][(pc_pr_[:, 4]>=slices[ind]) & (pc_pr_[:, 4]<=slices[ind+1])] = ind
    pc_[:, -1][(pc_[:, 8]>=slices[-1])] = len(slices) - 1
    pc_pr_[:, -1][(pc_pr_[:, 4]>=slices[-1])] = len(slices) - 1

    mean_energy = [np.mean(pc_[:, 9][pc_[:,-1]==ind]) for ind in range(len(slices))]
    std_energy = [np.std(pc_[:, 9][pc_[:,-1]==ind]) if pc_[:, 9][pc_[:,-1]==ind].shape[0] > 1 else None for ind in range(len(slices)) ]
    num_particles = [pc_[pc_[:,-1]==ind].shape[0] for ind in range(len(slices))]

    mean_energy_pred = [np.mean(pc_pr_[:, 5][pc_pr_[:,-1]==ind]) for ind in range(len(slices))]
    std_energy_pred = [np.std(pc_pr_[:, 5][pc_pr_[:,-1]==ind]) if pc_pr_[:, 5][pc_pr_[:,-1]==ind].shape[0] > 1 else None for ind in range(len(slices)) ]
    num_particles_pred = [pc_pr_[pc_pr_[:,-1]==ind].shape[0] for ind in range(len(slices))]

    axs[0,0].plot([slice_*1e3 for slice_ in slices], mean_energy, label='GT')
    axs[0,0].plot([slice_*1e3 for slice_ in slices], mean_energy_pred, label='Reconstructed')
    #axs[0,0].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[0,0].tick_params(axis='y', which='major', rotation=45)
    axs[0,0].grid(True)
    axs[0,0].set_xlabel('Z [mm]')
    axs[0,0].set_ylabel('Mean Energy [MeV]')
    axs[0,0].legend(prop={'size': 20})

    axs[0,1].plot([slice_*1e3 for slice_ in slices], num_particles, label='GT')
    axs[0,1].plot([slice_*1e3 for slice_ in slices], num_particles_pred, label='Reconstructed')
    #axs[0,1].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[0,1].tick_params(axis='y', which='major', rotation=45)
    axs[0,1].grid(True)
    axs[0,1].set_xlabel('Z [mm]')
    axs[0,1].set_ylabel('Number of particles')
    axs[0,1].legend(prop={'size': 20})

    axs[0,2].plot([slice_*1e3 for slice_ in slices], std_energy, label='GT')
    axs[0,2].plot([slice_*1e3 for slice_ in slices], std_energy_pred, label='Reconstructed')
    #axs[0,2].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[0,2].tick_params(axis='y', which='major', rotation=45)
    axs[0,2].grid(True)
    axs[0,2].set_xlabel('Z [mm]')
    axs[0,2].set_ylabel('Energy Deviation')
    axs[0,2].legend(prop={'size': 20})

    axs[1,0].scatter(pc[:,4]*1e3, pc[:,5], s=15, alpha=0.2, label='GT')
    axs[1,0].scatter(pc_pr[:,0]*1e3, pc_pr[:,1], s=15, alpha=0.2, label='Reconstructed')
    #axs[1,0].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[1,0].tick_params(axis='y', which='major', rotation=45)
    axs[1,0].grid(True)
    axs[1,0].set_xlabel('x [mm]')
    axs[1,0].set_ylabel('xp')
    axs[1,0].legend(prop={'size': 20})

    axs[1,1].scatter(pc[:,6]*1e3, pc[:,7], s=15, alpha=0.2, label='GT')
    axs[1,1].scatter(pc_pr[:,2]*1e3, pc_pr[:,3], s=15, alpha=0.2, label='Reconstructed')
    #axs[1,0].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[1,1].tick_params(axis='y', which='major', rotation=45)
    axs[1,1].grid(True)
    axs[1,1].set_xlabel('y [mm]')
    axs[1,1].set_ylabel('yp')
    axs[1,1].legend(prop={'size': 20})

    axs[1,2].scatter(pc[:,8]*1e3, pc[:,9], s=15, alpha=0.2, label='GT')
    axs[1,2].scatter(pc_pr[:,4]*1e3, pc_pr[:,5], s=15, alpha=0.2, label='Reconstructed')
    #axs[1,0].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[1,2].tick_params(axis='y', which='major', rotation=45)
    axs[1,2].grid(True)
    axs[1,2].set_xlabel('Z [mm]')
    axs[1,2].set_ylabel('Energy [MeV]')
    axs[1,2].legend(prop={'size': 20})

    axs[2,0].plot([slice_*1e3 for slice_ in slices], 
                  [np.std(pc_[:, 4][pc_[:,-1]==ind])*np.std(pc_[:, 5][pc_[:,-1]==ind])*1e3 for ind in range(len(slices))],
                 label='GT')
    axs[2,0].plot([slice_*1e3 for slice_ in slices], 
                  [np.std(pc_pr_[:, 0][pc_pr_[:,-1]==ind])*np.std(pc_pr_[:, 1][pc_pr_[:,-1]==ind])*1e3 for ind in range(len(slices))],
                 label='Reconstructed')

    #axs[2,0].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[2,0].tick_params(axis='y', which='major', rotation=45)
    axs[2,0].grid(True)
    axs[2,0].set_xlabel('Z [mm]')
    axs[2,0].set_ylabel('Stat Emmitance x [mm]')
    axs[2,0].legend(prop={'size': 20})

    axs[2,1].plot([slice_*1e3 for slice_ in slices], 
                  [np.std(pc_[:, 6][pc_[:,-1]==ind])*np.std(pc_[:, 7][pc_[:,-1]==ind])*1e3 for ind in range(len(slices))],
                 label='GT')
    axs[2,1].plot([slice_*1e3 for slice_ in slices], 
                  [np.std(pc_pr_[:, 2][pc_pr_[:,-1]==ind])*np.std(pc_pr_[:, 3][pc_pr_[:,-1]==ind])*1e3 for ind in range(len(slices))],
                 label='Reconstructed')
    #axs[2,1].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[2,1].tick_params(axis='y', which='major', rotation=45)
    axs[2,1].grid(True)
    axs[2,1].set_xlabel('Z [mm]')
    axs[2,1].set_ylabel('Stat Emmitance y [mm]')
    axs[2,1].legend(prop={'size': 20})

    axs[2,2].plot([slice_*1e3 for slice_ in slices], 
                  [np.std(pc_[:,8][pc_[:,-1]==ind])*np.std(pc_[:, 9][pc_[:,-1]==ind])*1e3 for ind in range(len(slices))],
                 label='GT')
    axs[2,2].plot([slice_*1e3 for slice_ in slices], 
                  [np.std(pc_pr_[:,4][pc_pr_[:,-1]==ind])*np.std(pc_pr_[:, 5][pc_pr_[:,-1]==ind])*1e3 for ind in range(len(slices))],
                 label='Reconstructed')

    #axs[2,2].xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    axs[2,2].tick_params(axis='y', which='major', rotation=45)
    axs[2,2].grid(True)
    axs[2,2].set_xlabel('Z [mm]')
    axs[2,2].set_ylabel('Stat Emmitance z [mm]')
    axs[2,2].legend(prop={'size': 20})

    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize']=20
    plt.rcParams['ytick.labelsize']=20

    kinds = ['lpa', 'Img1', 'Img2', 'Img4', 'UndEnt', 'UndMid', 'Img5', 'Img6']
    for kind in kinds:
        if kind in fname:
            name_=kind
            break
    name = (name_+'_emmit_x_' + str(pc[0,0]) + '_' +
                        'emmit_y_' + str(pc[0,1]) + '_' +
                        'beta_x_' + str(pc[0,2]) + '_' +
                        'beta_y_' + str(pc[0,3]))
    
    return fig, axs

def login_wandb(num_linear_layers=3,
                num_coupling_layers = 7,
                hidden_size = 128,
                learning_rate=1e-5):
    wandb_apikey = '...'
    wandb_project = '...'
    wandb_entity = '...'

    config_defaults = {
                'lr': learning_rate,
                'coupling_blocks': num_coupling_layers
            }

    os.environ['WANDB_API_KEY'] = wandb_apikey

    wandb.init(reinit=True, project=wandb_project, entity=wandb_entity, config=config_defaults)