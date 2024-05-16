# convert distributed model to single model
import torch
import torch.optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict

from common import AverageMeter, Transform
from dataset import AVADataset
from emd_loss import EDMLoss
from model import create_model
from scipy.stats import pearsonr, spearmanr


def is_ddp_ckpt(ckpt):
    sd = OrderedDict()
    for item in ckpt["state_dict"].items():
        if 'module.' not in item[0]:
            return (False, sd)
        sd_key = item[0][7:] #remove 'module.' 
        sd[sd_key] = item[1]
    return (True, sd)

def convert(src_file, dst_file):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_state = torch.load(src_file)
    
    #load weights to cpu
    ckpt = torch.load(src_file, map_location=torch.device("cpu")) 
    isddp, sd = is_ddp_ckpt(ckpt)
    if isddp:
        best_state["state_dict"] = sd
    else:
        pass

    torch.save(best_state, dst_file)

if '__main__' == __name__:
    convert('/data/wgl/fpg/best_state_dist.pth', '/data/wgl/fpg/best_state.pth')