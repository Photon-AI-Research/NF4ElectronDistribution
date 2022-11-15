import os
import sdds
import numpy as np

def load_pointcloud(path_in):
    '''
    Load files in SDDS format right after simulation
    '''
    #print('Extracting '+name+' from '+path_in)
    data = sdds.SDDS(0)
    data.load(path_in)

    ind_t = data.columnName.index('t')
    ind_p = data.columnName.index('p')
    ind_x = data.columnName.index('x')
    ind_xp = data.columnName.index('xp')
    ind_y = data.columnName.index('y')
    ind_yp = data.columnName.index('yp')
    pp = np.array(data.columnData[ind_p])[0]
    xx = np.array(data.columnData[ind_x])[0]
    xp = np.array(data.columnData[ind_xp])[0]
    zz = np.array(data.columnData[ind_y])[0]
    zp = np.array(data.columnData[ind_yp])[0]
    tt = np.array(data.columnData[ind_t])[0]
    #ss = 3e8*tt; ss-=np.mean(ss); ee = pp*0.511
    ss = tt
    ee = pp*0.511
    
    ps_data = np.stack((np.array(data.columnData[data.columnName.index('x')])[0],
                        np.array(data.columnData[data.columnName.index('xp')])[0],
                        np.array(data.columnData[data.columnName.index('y')])[0],
                        np.array(data.columnData[data.columnName.index('yp')])[0],
                        ss, ee),
                        #np.array(data.columnData[data.columnName.index('particleID')])[0]),
                        axis=-1)
    #extract parameters 
    pars = path_in.split("/")[-1].split('_')[1:]
    last_par = pars[-1].split('-')[0]+'-'+pars[-1].split('-')[1]
    pars_ = np.array([float(pars[0]), float(pars[1]), float(pars[2]), float(last_par)])

    all_ = np.concatenate((np.tile(pars_, (ps_data.shape[0], 1)),
                           ps_data), axis=1)

    return all_

def save_npys(files, path_out):
    '''
    Save files as numpy arrays for faster processing during training and validation of models
    '''
    for file in files:
        arr = load_pointcloud(file)
        filename = (file.split("/")[-1])[:-4]
        dir_name = '-'.join(filename.split('-')[:-1])
        if not os.path.exists(path_out + '/' + dir_name +'/'):
            os.makedirs(path_out + '/' + dir_name + '/')
        
        print('save ',filename+'.npy','\n\tto ',path_out+'/'+dir_name+ '/'+filename)
        np.save(path_out+'/'+dir_name+ '/'+filename+'.npy', arr)

dir_in = "./data/training_data"
dirs = [os.path.join(dir_in, o) for o in os.listdir(dir_in) 
                    if os.path.isdir(os.path.join(dir_in, o))]

files = []
for dir_ in dirs:
        #print('Process dir: ', dir_)
    for x in os.listdir(dir_):
        files.append(dir_ + '/' + x)
        
        
save_npys(files, './data/data_npy/training_data_npy')

dir_in = "./data/validation_data"
dirs = [os.path.join(dir_in, o) for o in os.listdir(dir_in) 
                    if os.path.isdir(os.path.join(dir_in, o))]

files = []
for dir_ in dirs:
        #print('Process dir: ', dir_)
    for x in os.listdir(dir_):
        files.append(dir_ + '/' + x)
        
        
save_npys(files, './data/data_npy/validation_data_npy')