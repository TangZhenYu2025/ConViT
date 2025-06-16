
from trian import train_model
from test import test_model
from dataset2 import train_loader
from dataset2 import test_loader
from dataset2 import valid_loader
import torch.optim as optim
import timm
import sklearn.metrics as sm
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from PIL import Image
import numpy as np
import sklearn.preprocessing as sp
import time
import sklearn.metrics as sm
import os
import torch
import torch.nn as nn

#CONVIT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_cfg=timm.create_model('convit_base').default_cfg
pretrained_cfg['file']='/tmp/pycharm_project_744/porous media/GradCAM/CNNKAN/pre_path_ok/convit_tiny.pth'
conv_layers = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256)
)
# 创建ConViT模型
model_convit = timm.create_model('convit_base', pretrained=False, num_classes=1, pretrained_cfg=pretrained_cfg, global_pool='avg')
# 组合卷积层和ConViT
model = nn.Sequential(
    conv_layers,
    model_convit
)
model[-1].head = nn.Sequential(
    nn.ReLU(),
    nn.BatchNorm1d(768),
    nn.Dropout(0.1),
    nn.Linear(in_features=768, out_features=1, bias=False)
)
print(model[-1].head)
model_name='ConViT'
blocks = model_convit.blocks
# print(blocks)
new_blocks = nn.Sequential(*list(blocks.children())[10:])
# 替换旧的 blocks 层
model_convit.blocks = new_blocks
print(model_convit)
convit_path = './'+ model_name+'_'+ str('dataszie=600,convit,number=6')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_convit = model_convit.to(device)

num_epochs = 200
optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
ce = torch.nn.HuberLoss()
train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
test_model(model_convit, convit_path, test_loader)
test_model(model_convit, convit_path, train_loader)



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
# swin_path = './'+ model_name+'_'+ str('200size')
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
# convit_path = './'+ model_name+'_'+ str('dataszie=200')
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
# optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)

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
# convit_path = './'+ model_name+'_'+ str('datasize=600')
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
# optimizer = optim.AdamW(model_convit.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# ce = torch.nn.HuberLoss()
# train_model(model_convit, convit_path, train_loader, valid_loader, num_epochs, optimizer, ce)
# test_model(model_convit, convit_path, test_loader)
# test_model(model_convit, convit_path, train_loader)

# resnet50
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