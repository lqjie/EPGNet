# The code is mainly borrowed from Remi COGRANNE, Thank Prof. COGRANNE
# EfficientNet code is based on https://github.com/lukemelas/EfficientNet-PyTorch
from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch
import os 
import random
import numpy as np
import argparse
import json
from dataset import *
from train import *
from models.enet_simple import get_model
import shutil

def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    '--config_path',
    help='config_path',
    type=str,
    )
    args = parser.parse_args()

    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_cuda_device(cuda_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_num)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #torch.backends.cudnn.deterministic = True

def main(args):
    SEED = 16121985
    seed_everything(SEED)

    with open(args.config_path,"r") as f:
        config = json.load(f)
    set_cuda_device(config["GPU"])
    device = torch.device('cuda')

    base_dir = f'./ckpt/{config["exp_name"]}'
    os.makedirs(base_dir, exist_ok=True)
    shutil.copy(args.config_path,base_dir)
    
    train_dataset, valid_dataset,test_dataset = get_train_valid_datasets_jpeg(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=config["batch_size"],
        pin_memory=False,
        drop_last=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=64,
        num_workers=0,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=64,
        num_workers=0,
        shuffle=False,
        sampler=SequentialSampler(test_dataset),
        pin_memory=False,
    )
    model = get_model(config["model_name"])
    model = model.to(device)
    iter_per_ep = len(train_loader)
    fitter = Fitter(model=model, device=device, config=config, n_iter_per_ep = iter_per_ep)
    if config["CURRICULUM"]:
        fitter.load(config["CL_path"])
        print('--> INITIAL <-- MODEL USED FOR TRAINING LOADED')

    fitter.fit(train_loader, val_loader, test_loader)

    fitter.load(f'{fitter.base_dir}/model-best.bin')
    print('--> TRAINED <-- MODEL WITH LOWEST LOSS LOADED')
    fitter.test(test_loader)

    fitter.load(f'{fitter.base_dir}/model-bestAcc.bin')
    print('--> TRAINED <-- MODEL WITH LOWEST ACCURACY LOADED')
    fitter.test(test_loader)

if __name__ == '__main__':
    args = myParseArgs()
    main(args)

