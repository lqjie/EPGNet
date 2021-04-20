from torch.utils.data import Dataset,DataLoader
import sklearn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import json
import torch
from scipy.io import loadmat
import numpy as np
import cv2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.0)

def get_valid_transforms():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], p=1.0)

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class Dataset_JPEG(Dataset):
    def __init__(self, data_list, transforms=None):
        super().__init__()
        self.data_list = data_list
        self.transforms = transforms
        self.labels = [data['label'] for data in data_list]
    def __getitem__(self, index: int):
        img_path, label = self.data_list[index]['img_path'], self.data_list[index]['label']
        image = loadmat(img_path,verify_compressed_data_integrity=False)['img']
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        target = onehot(2, label)
        return image, target
    def __len__(self) -> int:
        return len(self.data_list)
    def get_labels(self):
        return list(self.labels)

class Dataset_JPEG_SCA(Dataset):
    def __init__(self, data_list, transforms=None):
        super().__init__()
        self.data_list = data_list
        self.transforms = transforms
        self.labels = [data['label'] for data in data_list]
    def __getitem__(self, index: int):
        img_path, beta_path, label = self.data_list[index]['img_path'],self.data_list[index]['beta_path'],self.data_list[index]['label']

        image = loadmat(img_path,verify_compressed_data_integrity=False)['img']
        beta = loadmat(beta_path)['Beta']
        image = np.stack((image, beta), axis=-1)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        target = onehot(2, label)
        return image, target
    def __len__(self) -> int:
        return len(self.data_list)
    def get_labels(self):
        return list(self.labels)

def get_datalist(name_list, is_sca, cover_dir, stego_dir, cover_beta_dir, stego_beta_dir):
    if is_sca:
        perfix = [[cover_dir,cover_beta_dir], [stego_dir, stego_beta_dir]]
        dataset = []
        for name in name_list:
            for label, data_dir in enumerate(perfix):
                dataset.append({
                    'img_path': data_dir[0]+name.replace('tif','mat'),
                    'beta_path': data_dir[1]+name.replace('tif','mat'),
                    'label': label
                })

    else:
        perfix = [cover_dir, stego_dir]
        dataset = []
        for name in name_list:
            for label, data_dir in enumerate(perfix):
                dataset.append({
                    'img_path': data_dir+name.replace('tif','mat'),
                    'label': label
                })

    return dataset

def get_train_valid_datasets_jpeg(config):
    with open(config['data_split'] +'.json',"r") as f:
        data_names = json.load(f)
    train_names = data_names['train']
    valid_names = data_names['valid']
    test_names = data_names['test']

    # The format of data_split.json is similar to the following:
    # {"train": ["20690.tif", "65494.tif", ..],
    #  "valid": ["49460.tif", "18854.tif", ...],
    #  "test": ["67702.tif", "18080.tif", ...]}

    train_list = get_datalist(train_names, config['is_sca'], config['cover_dir'], config['stego_dir'], config['cover_beta_dir'], config['stego_beta_dir'])
    valid_list = get_datalist(valid_names, config['is_sca'], config['cover_dir'], config['stego_dir'], config['cover_beta_dir'], config['stego_beta_dir'])
    test_list = get_datalist(test_names, config['is_sca'], config['cover_dir'], config['stego_dir'], config['cover_beta_dir'], config['stego_beta_dir'])

    if config['is_sca']:

        train_dataset = Dataset_JPEG_SCA(
            data_list=train_list,
            transforms=get_train_transforms(),
        )

        valid_dataset = Dataset_JPEG_SCA(
            data_list=valid_list,
            transforms=get_valid_transforms(),
        )
        test_dataset = Dataset_JPEG_SCA(
            data_list=test_list,
            transforms=get_valid_transforms(),
        )
    else:
        train_dataset = Dataset_JPEG(
            data_list=train_list,
            transforms=get_train_transforms(),
        )

        valid_dataset = Dataset_JPEG(
            data_list=valid_list,
            transforms=get_valid_transforms(),
        )
        test_dataset = Dataset_JPEG(
            data_list=test_list,
            transforms=get_valid_transforms(),
        )
    print( "Number of images for TRAIN - VAL - TEST = " , len(train_dataset) , " - " , len(valid_dataset) , " - " , len(test_dataset) )
    return train_dataset, valid_dataset,test_dataset
