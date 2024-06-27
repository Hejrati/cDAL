import random

from torch import nn
from torch.backends import cudnn
import torch
import shutil
import os
import numpy as np
import torch.distributed as dist


# %%

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def set_random_seed_for_iterations(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def dev(gpu):
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    return torch.device("cpu")

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname.find("DownConv") == -1 and classname.find("UpConv") == -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class CustomDDPWrapper(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)