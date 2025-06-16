
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


import sklearn.metrics as sm
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
# import cv2
from PIL import Image
import numpy as np
import sklearn.preprocessing as sp
import time
import sklearn.metrics as sm
import os
import torch
import os
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
class Mydata(Dataset):
    def __init__(self, path, target, if_train=True,if_valid=True, transform=None):
        if if_train and if_valid:
            self.df = pd.read_csv(path + 'permeability.csv')
            self.img_path = "porous_media_images"
        else:
            self.df = pd.read_csv(path + 'rock_perm.csv')
            self.img_path = "rock images"
        self.target = target
        self.path = path
        self.transform = transform

        # 计算并应用归一化
        self.scaler = MinMaxScaler()
        self.df[self.target] = self.scaler.fit_transform(self.df[[self.target]])

    def __getitem__(self, index):
        i_data = self.df.iloc[index]
        img_path = self.path + self.img_path + '/' + str(index + 1) + ".png"
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(i_data[self.target], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.df)

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.RandomCrop(224, padding=4),
transforms.ToTensor(),
transforms.Normalize(norm_mean, norm_std)
])
# 6:2:2
dataset = Mydata('../2Dporousmediaimages/', 'permeability (mD)',transform=transform)
total_count = dataset.__len__()
test_count = int(0.15* total_count)
valid_count = int(0.15* total_count)
train_count = total_count - test_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, valid_count, test_count), generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
print('Total: {}, Train: {}, Vali: {}, Test: {}'.format(total_count,train_count,valid_count, test_count))