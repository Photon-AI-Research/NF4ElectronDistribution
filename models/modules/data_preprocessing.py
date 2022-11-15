import os
import random
import numpy as np
import torch
import wandb

def prepare_paths(dir_in):
    '''
    Create list of paths to all files from simulations in dir_in(string)
    '''
    dirs = [os.path.join(dir_in, o) for o in os.listdir(dir_in) 
                    if os.path.isdir(os.path.join(dir_in, o))]

    files = []
    for dir_ in dirs:
        for x in os.listdir(dir_):
            files.append(dir_ + '/' + x)
    return files

def get_data(ind, 
             items,
             positions,
             num_elec=-1,
             time_stamp=8,
             num_inputs=1):
    '''
    Load file with 1 electron cloud
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        num_elec(integer): number of electrons to sample from an electron cloud, 
                           if -1 then take electron cloud completely
        time_stamp(integer): index of column that corresponds to time stamp
        positions(list of string): suffixes of files that correspond to positions in beamline, where electron clouds were observed
        num_inputs(integer): number of input variables, that start from index 0 until num_inputs-1 in dimension 2
                           
    Each file is a 2D numpy array with N raws(corresponds to N electrons)
    '''
    
    pointcloud = np.load(items[ind])
    
    #convert time to a spacial coordinate and center around 0
    pointcloud[:, time_stamp] = pointcloud[:, time_stamp]*3e8 - np.mean(pointcloud[:, time_stamp]*3e8)
    #add hot vector that corresponds to a positions in the beamline
    pointcloud = extend_to_hotvec(items[ind], pointcloud, positions, num_inputs)

    arr = torch.from_numpy(pointcloud).float()
    if num_elec == -1:
        return arr
    else:
        inds = random.sample(list(range(0, arr.shape[0])), num_elec)
        return arr[inds, :]
    
def extend_to_hotvec(item,
                       arr,
                       positions,
                       num_inputs):
    '''
    Extends each raw of simulation by a hot vector that corresponds to a position in a beamline
    Args:
        item(string): filename of one simulation file(numpy array with an electron cloud)
        arr(2D numpy array): array of electrons with their phase space coordinates (1st dimension: number of electron, 2nd: phase space coodinate)
        positions(list of string): suffixes of files that correspond to positions in beamline, where electron clouds were observed
        num_inputs(integer): number of input variables, that start from index 0 until num_inputs-1 in dimension 2
    
    Example: 
    item="*Imager2*.npy"
    positions=["Imager1", "Imager2", "Imager3"], then each filename has to contain one of these suffixes   
    if arr has the structure for each particle(raw) [input1, input2, x, xp, en]
    then function will extend it for each particle to be [input1, input2, 0., 1., 0., x, xp, en]
    '''
    
    for ind_, position in enumerate(positions):
        if position in item:
            a = np.full((1, arr.shape[0]), ind_)
            a = np.concatenate((a, np.array([[len(positions)-1]])), axis=1)
            b = np.zeros((a.size, a.max()+1))
            b[np.arange(a.size),a] = 1
            arr_ = np.insert(arr, [num_inputs for i in range(len(positions))], b[:-1,:], axis=1)
    return arr_

def normalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Normalize point from a set of points with vmin(minimum) and vmax(maximum)
    to be in a range [a, b]
    '''
    return (a + (point - vmin) * (b - a) / ( vmax - vmin))

def denormalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Denormalize point from range [a, b]
    to be in set of points with vmin(minimum) and vmax(maximum)
    '''
    return ((point - a) * (vmax - vmin) / (b - a) + vmin)

def get_vmin_vmax(items, positions, time_stamp, num_inputs):
    '''
    Find minima/maxima in all columns among a complete data(all simulation files)
    Args:
        items(list of string): list of paths to all electron clouds
        positions(list of string): suffixes of files that correspond to positions in beamline, where electron clouds were observed
        time_stamp(integer): index of column that corresponds to time stamp
        num_inputs(integer): number of input variables, that start from index 0 until num_inputs-1 in dimension 2
    
    returns torch tensors with minima and maxima
    '''
    
    for item in items:
        arr = np.load(item)

        #convert time to a spacial coordinate and center around 0
        arr[:, time_stamp] = arr[:, time_stamp]*3e8 - np.mean(arr[:, time_stamp]*3e8)
        arr = extend_to_hotvec(item, arr, positions, num_inputs)

        if item == items[0]:
            vmin = [np.min(arr[:, i]) for i in range(arr.shape[1])]
            vmax = [np.max(arr[:, i]) for i in range(arr.shape[1])]
        else:
            vmin = [min(np.min(arr[:, i]), vmin[i]) for i in range(arr.shape[1])]
            vmax = [max(np.max(arr[:, i]), vmax[i]) for i in range(arr.shape[1])]
    return torch.Tensor(vmin), torch.Tensor(vmax)