import os
import random
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

from MANIQA.config import config
from MANIQA.util.options import sobel


class IQA_datset(torch.utils.data.Dataset):
    def __init__(self, config, scene_list, transform, mode='train'):
        super(IQA_datset, self).__init__()
        self.config = config
        self.scene_list = scene_list
        self.transform = transform
        self.mode = mode

        self.dis_path = self.config.db_path
        self.txt_file_name = self.config.text_path

        idx_data, dis_files_data, score_data = [], [], []

        sai_list_sel = [
            '046', '047', '048',
            '049', '050', '051',
            '052', '053', '054',
        ]
        epi_list_x = ['0']
        epi_list_y = ['0']

        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                idx, dis, score = line.split()
                idx = int(idx)
                score = float(score)

                if idx in self.scene_list:
                    sai_list, epi_list = [], []

                    for i in range(len(sai_list_sel)):  # 4

                        sai_each = '{}/Frame_{}.png'.format(dis, sai_list_sel[i])
                        sai_list.append(sai_each)

                    epi_x = '{}/{}.png'.format(dis, epi_list_x[0])
                    epi_y = '{}/{}.png'.format(dis, epi_list_y[0])
                    epi_list.append(epi_x)
                    epi_list.append(epi_y)

                    dis_files_data.append([sai_list, epi_list])
                    idx_data.append(idx)
                    score_data.append(score)

        # reshape list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        idx_data = np.array(idx_data)
        idx_data = idx_data.reshape(-1, 1)

        self.data_dict = {
            'lf_list': dis_files_data,
            'score_list': score_data,
            'idx_list': idx_data
        }

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['lf_list'])

    def __getitem__(self, idx):
        h, w = 720, 960
        crop_h, crop_w = 512, 512
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        bottom = top + crop_h
        right = left + crop_w

        if_flip = random.random()  # 随机数

        sai_list, epi_list = [], []

        '''sai'''
        for i in range(len(self.data_dict['lf_list'][idx][0])):

            sai_each = self.data_dict['lf_list'][idx][0][i]
            # cv
            sai = cv2.imread(os.path.join(self.dis_path, sai_each), cv2.IMREAD_COLOR)
            sai = cv2.cvtColor(sai, cv2.COLOR_BGR2RGB)
            sai = sai[top:bottom, left:right]

            # sai = sobel(sai)  # Sobel

            if if_flip > 0.5 and self.mode == 'train':
                sai = cv2.flip(sai, 1)

            # PIL
            # sai = Image.open(Path(self.config.db_path) / sai_each).convert("RGB")
            # # sai = sai.resize(self.config.input_size)
            # if if_flip > 0.5 and self.mode == 'train':
            #     sai = sai.transpose(Image.FLIP_LEFT_RIGHT)
            # # sai = sai.crop((left, top, right, bottom))

            if self.transform:
                sai = self.transform(sai)
            sai_list.append(sai)

        '''epi'''
        for i in range(len(self.data_dict['lf_list'][idx][1])):
            epi_each = self.data_dict['lf_list'][idx][1][i]
            # cv
            epi = cv2.imread(os.path.join(self.config.epi_path, epi_each), cv2.IMREAD_COLOR)
            epi = cv2.cvtColor(epi, cv2.COLOR_BGR2RGB)
            if i == 0:
                epi = epi[0:32, 0:512]
            else:
                epi = epi[101-32:101, 960-512:960]

            epi = sobel(epi)  # Sobel

            # PIL
            # epi = Image.open(Path(self.config.epi_path) / epi_each).convert("RGB")
            # if i == 0:
            #     epi = epi.crop((0, 0, 512, 32))
            # else:
            #     epi = epi.crop((960-512, 101-32, 960, 101))

            if self.transform:
                epi = self.transform(epi)
            epi_list.append(epi)

        score = self.data_dict['score_list'][idx]
        idx = self.data_dict['idx_list'][idx]
        sample = {
            'lf_list': [sai_list, epi_list],
            'score': score,
            'idx': idx
        }

        return sample
