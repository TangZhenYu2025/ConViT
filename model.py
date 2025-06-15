# 人员：汤振宇
# 开发时间: 2024/8/29 18:09
from trian import train_model
from test import test_model
from dataset2 import train_loader
from dataset2 import test_loader
from dataset2 import valid_loader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import timm

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
from demo11 import ResNet50
from resnet_new import resnet50

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = 'cuda:1'
# # 创建简单的CNN模型实例
# model_resnet50 = resnet50()
# torch.cuda.set_device('cuda:{}'.format(1))
# print(model_resnet50)
#
# # 将模型移动到指定设备上
# model_resnet50.to(device)
# model_name = 'resnet50'
# resnet50_path = './' + model_name+'_'+ str('datasize=200')
# path = os.path.join(resnet50_path, 'best.pth')
# if os.path.exists(path):
#     resnet50_path.load_state_dict(torch.load(path))
#     print("model load successful")
# # 定义优化器
# optimizer_resnet50 = optim.Adam(model_resnet50.parameters(), lr=0.001, weight_decay=1e-2)
# # 定义学习率调度器
# scheduler_resnet50 = optim.lr_scheduler.ExponentialLR(optimizer_resnet50, gamma=0.9)
# # 损失参数
# # ce = torch.nn.MSELoss()
# ce = torch.nn.HuberLoss()
# # 迭代次数
# num_epochs = 200
#
# train_model(model_resnet50, resnet50_path, train_loader, valid_loader, num_epochs, optimizer_resnet50, ce)
# test_model(model_resnet50, resnet50_path, test_loader)
# test_model(model_resnet50,resnet50_path, train_loader)


#CONVIT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_cfg=timm.create_model('convit_base').default_cfg
pretrained_cfg['file']='/tmp/pycharm_project_744/porous media/GradCAM/CNNKAN/pre_path_ok/convit_tiny.pth'
model_convit = timm.create_model('convit_base', pretrained=False, num_classes=1,pretrained_cfg=pretrained_cfg,global_pool='avg')
model_name='convit_base'
# pre_path = '/tmp/pycharm_project_744/porous media/GradCAM/CNNKAN/pre_path_ok/convit_base.pth'
# model_convit = timm.create_model(model_name, pretrained=True, num_classes=1000,pretrained_cfg=pretrained_cfg,global_pool='avg')
model_convit.head = nn.Sequential(
    nn.ReLU(),
    nn.BatchNorm1d(768),
    nn.Dropout(0.1),
    nn.Linear(in_features=768, out_features=1, bias=False),
    )


print(model_convit.head)
# print(model_convit)
# from torchsummary import summary
# model_convit = model_convit.to('cuda')
# # 假设 model_convit 是你已经定义的模型
# summary(model_convit, input_size=(3, 224, 224))  # 修改 input_size 以匹配模型输入
blocks = model_convit.blocks
# print(blocks)
new_blocks = nn.Sequential(*list(blocks.children())[10:])
# 替换旧的 blocks 层
model_convit.blocks = new_blocks
print(model_convit)
convit_path = './'+ model_name+'_'+ str('dataszie=600,convit,number=6')

# path = os.path.join(convit_path, 'best.pth')
# if os.path.exists(path):
#     model_convit.load_state_dict(torch.load(path))
#     print("model load successful")
# #%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_device('cuda:{}'.format(0))
model_convit = model_convit.to(device)

num_epochs = 200
optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
ce = torch.nn.HuberLoss()
train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
test_model(model_convit, convit_path, test_loader)
test_model(model_convit, convit_path, train_loader)




import collections
## botnet
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_resnet50 = ResNet50()
# torch.cuda.set_device('cuda:{}'.format(1))
# print(model_resnet50)
#
# # 将模型移动到指定设备上
# model_resnet50.to(device)
# model_name = 'resnet50'
# resnet50_path = './' + model_name+'_'+ str('test_count')
# # path = os.path.join(resnet50_path, 'best.pth')
# # if os.path.exists(path):
# #     resnet50_path.load_state_dict(torch.load(path))
# #     print("model load successful")
# # 定义优化器
# optimizer_resnet50 = optim.Adam(model_resnet50.parameters(), lr=0.01, weight_decay=1e-2)
# # 定义学习率调度器
# scheduler_resnet50 = optim.lr_scheduler.ExponentialLR(optimizer_resnet50, gamma=0.9)
# # 损失参数
# # ce = torch.nn.MSELoss()
# ce = torch.nn.HuberLoss()
# # 迭代次数
# num_epochs = 300
#
# train_model(model_resnet50, resnet50_path, train_loader, valid_loader, num_epochs, optimizer_resnet50, ce)
# test_model(model_resnet50, resnet50_path, test_loader)
# test_model(model_resnet50,resnet50_path, train_loader)

## Twins_svt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name='twins_svt_small.in1k'
# model = timm.create_model(model_name, pretrained=False,num_classes=1, global_pool='avg')
# model.head = nn.Sequential(
#     nn.ReLU(),
#     nn.BatchNorm1d(512),
#     nn.Dropout(0.1),
#     nn.Linear(in_features=512, out_features=1, bias=False),
#     )
# convit_path = './'+ model_name+'_'+ str(400)
#
# # path = os.path.join(convit_path, 'best.pth')
# # if os.path.exists(path):
# #     model_convit.load_state_dict(torch.load(path))
# #     print("model load successful")
# #%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device('cuda:{}'.format(0))
# model_convit = model.to(device)
#
# num_epochs = 300
# optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)

#Twins_pcpvt_small
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name='twins_pcpvt_small.in1k'
# model = timm.create_model(model_name, pretrained=False,num_classes=1, global_pool='avg')
# model.head = nn.Sequential(
#     nn.ReLU(),
#     nn.BatchNorm1d(512),
#     nn.Dropout(0.1),
#     nn.Linear(in_features=512, out_features=1, bias=False),
#     )
# convit_path = './'+ model_name+'_'+ str(400)
#
# # path = os.path.join(convit_path, 'best.pth')
# # if os.path.exists(path):
# #     model_convit.load_state_dict(torch.load(path))
# #     print("model load successful")
# #%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device('cuda:{}'.format(0))
# model_convit = model.to(device)
#
# num_epochs = 100
# optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)

# ### twins_pcpvt_base
# model_name='twins_pcpvt_base.in1k'
# model = timm.create_model(model_name, pretrained=False,num_classes=1, global_pool='avg')
# model.head = nn.Sequential(
#     nn.ReLU(),
#     nn.BatchNorm1d(512),
#     nn.Dropout(0.1),
#     nn.Linear(in_features=512, out_features=1, bias=False),
#     )
# convit_path = './'+ model_name+'_'+ str(400)
#
# # path = os.path.join(convit_path, 'best.pth')
# # if os.path.exists(path):
# #     model_convit.load_state_dict(torch.load(path))
# #     print("model load successful")
# #%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device('cuda:{}'.format(0))
# model_convit = model.to(device)
#
# num_epochs = 300
# optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)

# convnext
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model_name='swin_tiny_patch4_window7_224.ms_in1k'
# model_name='convnext_small.fb_in22k_ft_in1k'
# model = timm.create_model(model_name, pretrained=False,num_classes=1, global_pool='avg')
# # model.head = nn.Sequential(
# #     nn.ReLU(),
# #     nn.BatchNorm2d(768),
# #     nn.Dropout(0.1),
# #     nn.Linear(in_features=768, out_features=1, bias=False),
# #     )
# new_fc = torch.nn.Sequential(collections.OrderedDict([
#     ('relu', torch.nn.ReLU()),# GELU ReLU
#     ('dropout', torch.nn.Dropout(0.1)),
#     ('fc', torch.nn.Linear(768, 1))
#     ]))
#
# model.head.fc = new_fc
# convit_path = './'+ model_name+'_'+ str(400)
#
# # path = os.path.join(convit_path, 'best.pth')
# # if os.path.exists(path):
# #     model_convit.load_state_dict(torch.load(path))
# #     print("model load successful")
# #%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device('cuda:{}'.format(0))
# model_convit = model.to(device)
#
# num_epochs = 200
# optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)

# VIT
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name='vit_base_patch16_18x2_224'
# model = timm.create_model(model_name, pretrained=False,num_classes=1, global_pool='avg')
# model.head = nn.Sequential(
#     nn.ReLU(),
#     nn.BatchNorm1d(768),
#     nn.Dropout(0.1),
#     nn.Linear(in_features=768, out_features=1, bias=False),
#     )
# convit_path = './'+ model_name+'_'+ str('400size')
# print(model)
# # path = os.path.join(convit_path, 'best.pth')
# # if os.path.exists(path):
# #     model_convit.load_state_dict(torch.load(path))
# #     print("model load successful")
# #%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device('cuda:{}'.format(0))
# model_convit = model.to(device)
# model_name = 'vit'
# vit_path = './' + model_name+'_'+ str('结果2')
# path = os.path.join(vit_path, 'best.pth')
# if os.path.exists(path):
#     vit_path.load_state_dict(torch.load(path))
#     print("model load successful")
# num_epochs = 200
# optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)

# swim-transformer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = 'swin_base_patch4_window7_224'
# pretrained_cfg=timm.create_model('swin_base_patch4_window7_224').default_cfg
# pretrained_cfg['file']='/tmp/pycharm_project_744/porous media/GradCAM/CNNKAN/pre_path_ok/swin_base_patch4_window7_224_22kto1k.pth'
# model_swin= timm.create_model(model_name, pretrained=False, num_classes=1,global_pool='avg',pretrained_cfg=pretrained_cfg)
# o = model_swin(torch.randn(2, 3, 224, 224))
# # model_swin = timm.create_model(model_name, pretrained=False, num_classes=1, global_pool='avg')
# # print(model_swin.default_cfg)
# # pre_path = 'C:/Python/porous media/GradCAM/CNNKAN/pre_path_ok/swin_base_patch4_window7_224_22kto1k.pth'
#
#
# print(model_swin.head)
# new_fc = torch.nn.Sequential(collections.OrderedDict([
#     ('relu', torch.nn.ReLU()),# GELU ReLU
#     ('dropout', torch.nn.Dropout(0.1)),
#     ('fc', torch.nn.Linear(1024, 1))
#     ]))
#
# model_swin.head.fc = new_fc
# print(model_swin.head)
#
# swin_path = './'+ model_name+'_'+ str('400size')
#
# # path = os.path.join(swin_path, 'best.pth')
# # if os.path.exists(path):
# #     model_swin.load_state_dict(torch.load(path))
# #     print("model load successful")
# #%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device('cuda:{}'.format(0))
# model_swin = model_swin.to(device)
#
# num_epochs =200
# optimizer = torch.optim.Adam(model_swin.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#
# ce = torch.nn.HuberLoss()
#
#
# #%%
# train_model(model_swin, swin_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# #%%
# test_model(model_swin, swin_path, test_loader)
# #%%
# test_model(model_swin, swin_path, train_loader)

# densenet
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name='densenet121'
# model = timm.create_model(model_name, pretrained=False,num_classes=1, global_pool='avg')
# model.head = nn.Sequential(
#     nn.ReLU(),
#     # nn.BatchNorm1d(1024),
#     nn.Dropout(0.1),
#     nn.Linear(in_features=1024, out_features=1, bias=False),
#     )
# convit_path = './'+ model_name+'_'+ str('datasize=200')
# print(model)
# # path = os.path.join(convit_path, 'best.pth')
# # if os.path.exists(path):
# #     model_convit.load_state_dict(torch.load(path))
# #     print("model load successful")
# #%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device('cuda:{}'.format(0))
# model_convit = model.to(device)
# num_epochs = 200
# optimizer = optim.Adam(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)