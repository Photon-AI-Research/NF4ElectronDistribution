import torch
from torch.utils.data import Dataset
import data_preprocessing

class PCDataset(Dataset):
    def __init__(self, 
                 items, 
                 positions,
                 num_points=-1,
                 num_files=1,
                 time_stamp=8,
                 num_inputs=4,
                 normalize=False, a=0., b=1.):
        '''
        Prepare dataset
        Args:
            items(list of string): list of paths to files with simulations
            positions(list of string): suffixes of files that correspond to positions in beamline, where electron clouds were observed
            num_points(integer): number of points to sample from each electron cloud, if -1 then take a complete electron cloud 
            num_files(integer): number of files to take for a dataset
            time_stamp(integer): index of column that corresponds to time stamp
            num_inputs(integer): number of input variables, that start from index 0 until num_inputs-1 in dimension 2
            normalize(boolean): True if normalize each point to be in range [a, b]
        '''

        self.get_data = data_preprocessing.get_data
        self.normalize = normalize
        self.num_points = num_points
        self.time_stamp = time_stamp
        self.num_inputs = num_inputs
        self.positions = positions
        self.a, self.b = a, b 
        
        self.vmin, self.vmax = data_preprocessing.get_vmin_vmax(items, positions, time_stamp, num_inputs)

        if num_files == -1:
            self.items = items
        else:
            self.items = items[0:num_files]
            for t in self.items:
                print(t.split('/')[-1])

        self.num_files = len(self.items)
        
        #print('Number of imagers: ', len(positions))
        #print('Total number of files: ', len(self.items))

    def __getitem__(self, index):
        return self.get_data(index, 
                             items=self.items,
                             positions=self.positions,
                             num_elec=self.num_points,
                             time_stamp=self.time_stamp,
                             num_inputs=self.num_inputs)

    def __len__(self):
        return self.num_files