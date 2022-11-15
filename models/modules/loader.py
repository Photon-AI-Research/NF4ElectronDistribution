from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
import torch

def get_loader(dataset, batch_size=1, use_hvd=False):
    if len(dataset) == 0:
         raise ValueError('Loader is empty')
    if use_hvd:
        import horovod.torch as hvd
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        sampler = RandomSampler(list(range(len(dataset)))) 
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print('Total number of points: ', dataset.__len__())
    print('Size of loader: ', len(loader))
    return loader