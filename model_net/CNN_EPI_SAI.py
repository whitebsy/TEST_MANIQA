import random

import torch
import torch.nn as nn
import timm
# import torch.nn.functional as F
import torchvision.transforms.functional as F
from torch import nn
from einops import rearrange
from torchvision import models

from model_net.resnet_backbone import resnet34_backbone, resnet50_backbone


class LFIQA(nn.Module):
    def __init__(self, drop=0.2):
        super().__init__()

        self.res50 = models.resnet50(pretrained=True)
        # self.res50.conv1 = nn.Conv2d(9, 64, 7, 2, 3)
        self.res50 = torch.nn.Sequential(*list(self.res50.children())[:-1])  # 去除最后的层数,cat输入时只去除fc层
        self.resnet34 = resnet34_backbone('/home/lin/Work/MANIQA-master/MANIQA/models/resnet34.pth')
        self.resnet50 = resnet50_backbone('/home/lin/Work/MANIQA-master/MANIQA/models/resnet50.pth')
        # self.resnet50.conv1 = nn.Conv2d(9, 64, 7, 2, 3)

        self.res34 = models.resnet34(pretrained=True)
        # self.res34.conv1 = nn.Conv2d(9, 64, 7, 2, 3)
        self.res34 = torch.nn.Sequential(*list(self.res34.children())[:-1])  # 去除最后的层数,fc,avg_pool

        self.conv6144 = nn.Conv2d(6144, 2048, 1, 1, 0)
        self.conv4608 = nn.Conv2d(4608, 1024, 1, 1, 0)
        self.conv2048 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.conv1024 = nn.Conv2d(1024, 512, 1, 1, 0)

        self.CNN_Block = nn.Sequential(
            # 1
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # 4
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # 5
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_num = 256
        self.fb_num = 512
        self.fc_num = 1536  # 1024, 1280,1536,2560
        self.resb_num = 2048  # 512, 2048
        self.CNN_Block2 = nn.Sequential(
            nn.Linear(self.cnn_num, self.cnn_num),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(self.cnn_num, self.cnn_num),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.sai_block = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

        self.Fb = nn.Sequential(
            nn.ReLU(),

            nn.Linear(self.fb_num, self.fb_num),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(self.fb_num, self.fb_num),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.res_block = nn.Sequential(
            nn.Linear(self.resb_num, self.resb_num),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(self.resb_num, self.resb_num // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),

            nn.Linear(self.resb_num // 2, self.resb_num // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.Sc = nn.Sequential(
            nn.Linear(self.fc_num, self.fc_num),
            # nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(self.fc_num, 1),
        )

        self.avgpool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2048 = nn.BatchNorm2d(2048)
        self.bn1024 = nn.BatchNorm2d(1024)
        self.bn512 = nn.BatchNorm2d(512)
        self.elu = nn.ELU()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [b, 9, 224, 224]

        sai, epi = x

        '''EPI'''
        input_epi = True
        if input_epi:
            x, y = epi
            rand_num = random.randint(1, 512 - 16)
            x = x[:, :, :9, rand_num:rand_num+16]
            # y = y[:, :, rand_num:rand_num+16, :]
            y = y[:, :, :9, rand_num:rand_num+16]

            x = self.CNN_Block(x)  # [4, 128, 1, 16]
            y = self.CNN_Block(y)  # [4, 128, 16, 1]
            x = torch.flatten(x, 1)
            y = torch.flatten(y, 1)
            # x = x[:, :, 0, 0]
            # y = y[:, :, 0, 0]
            x = self.CNN_Block2(x)
            y = self.CNN_Block2(y)
            e = torch.concat((x, y), dim=1)
            # e = self.elu(e)
            # e = self.relu(e)
            # e = torch.flatten(e, 1)
            e = self.Fb(e)

        '''SAI'''
        l = []

        # 1
        # # for i in range(len(sai)):
        # #     sai[i] = F.rgb_to_grayscale(sai[i])
        # #     sai[i] = torch.unsqueeze(sai[i], 1)
        #
        # # s = torch.concat(sai, dim=1)
        # s = torch.concat((sai[0], sai[1], sai[2]), dim=1)
        # s = self.res50(s)
        # s = self.conv2048(s)
        # # s = self.conv1024(s)
        # # s = self.avgpool_2d(s)
        # s = torch.flatten(s, 1)
        # # s = s[:, :, 0, 0]
        # # s = self.res_block(s)

        # 2
        # s1 = torch.concat((sai[0], sai[1], sai[2]), dim=1)
        # s2 = torch.concat((sai[3], sai[4], sai[5]), dim=1)
        # s3 = torch.concat((sai[6], sai[7], sai[8]), dim=1)
        # s1 = self.res50(s1)
        # s2 = self.res50(s2)
        # s3 = self.res50(s3)
        # s = torch.concat((s1, s2, s3), dim=1)
        # s = self.conv6144(s)

        # 3
        for i in range(len(sai)):
            f = self.res34(sai[i])
            # f = self.res50(sai[i])
            # f = self.conv2048(f)
            l.append(f)
        s = torch.concat(l, dim=1)
        s = self.conv4608(s)
        # s = self.conv1024(s)
        # s = self.avgpool_2d(s)
        s = torch.flatten(s, 1)
        # # s = self.sai_block(s)
        # # s = torch.squeeze(s)

        if input_epi:
            score = torch.cat((e, s), dim=1)  # !!
            score = self.Sc(score)
        else:
            score = self.Sc(s)
        return score
