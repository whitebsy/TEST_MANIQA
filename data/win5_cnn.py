import os
import random
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

from MANIQA.config import config


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
            '4_4', '4_5', '4_6',
            '5_4', '5_5', '5_6',
            '6_4', '6_5', '6_6',
        ]
        epi_list_x = ['x_5']
        epi_list_y = ['y_5']

        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                idx, dis, score = line.split()
                idx = int(idx)
                score = float(score)

                if idx in self.scene_list:
                    sai_list, epi_list = [], []

                    for i in range(len(sai_list_sel)):  # 4

                        sai_each = '{}/{}.png'.format(dis, sai_list_sel[i])
                        sai_list.append(sai_each)

                    epi_x = '{}/{}.png'.format(dis, epi_list_x[0])
                    epi_y = '{}/{}.png'.format(dis, epi_list_y[0])
                    epi_list.append(epi_x)
                    epi_list.append(epi_y)

                    dis_files_data.append([sai_list, epi_list])
                    idx_data.append(idx)
                    score_data.append(score)
                    # print(idx, '\t', sai_list, '\t', score)
                    # exit()

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
        h, w = self.config.input_size
        top = random.randint(0, h - config.crop_size)
        left = random.randint(0, w - config.crop_size)
        bottom = top + config.crop_size
        right = left + config.crop_size

        if_flip = random.random()  # 随机数

        sai_list, epi_list = [], []
        '''1-'''
        # for i in range(len(self.data_dict['sai_list'][idx])):
        #
        #     sai_each = self.data_dict['sai_list'][idx][i]
        #     sai = Image.open(Path(self.config.db_path) / sai_each).convert("RGB")
        #
        #     sai = sai.resize(self.config.input_size)
        #
        #     '''翻转、裁剪'''
        #     if if_flip > 0.5 and self.mode == 'train':
        #         sai = sai.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转, ++
        #
        #     sai = sai.crop((left, top, right, bottom))
        #
        #     if self.transform:
        #         sai = self.transform(sai)
        #     sai_list.append(sai)

        '''sai'''
        for i in range(len(self.data_dict['lf_list'][idx][0])):

            sai_each = self.data_dict['lf_list'][idx][0][i]
            sai = cv2.imread(os.path.join(self.config.db_path, sai_each), cv2.IMREAD_COLOR)
            sai = cv2.cvtColor(sai, cv2.COLOR_BGR2RGB)
            # sai = cv2.cvtColor(sai, cv2.COLOR_BGR2GRAY)
            sai = cv2.resize(sai, (512, 512))
            # sai = sai[top:bottom, left:right]

            # Sobel
            x = cv2.Sobel(sai, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(sai, cv2.CV_16S, 0, 1)
            abs_x = cv2.convertScaleAbs(x)
            abs_y = cv2.convertScaleAbs(y)
            dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
            sai = dst                                         # ////////

            if if_flip > 0.5 and self.mode == 'train':
                sai = cv2.flip(sai, 1)
            sai = np.array(sai).astype('float32') / 255
            sai = np.transpose(sai, (2, 0, 1))

            if self.transform:
                sai = self.transform(sai)
            sai_list.append(sai)

        '''epi'''
        for i in range(len(self.data_dict['lf_list'][idx][1])):
            epi_each = self.data_dict['lf_list'][idx][1][i]
            epi = cv2.imread(os.path.join(self.config.epi_path, epi_each), cv2.IMREAD_COLOR)
            epi = cv2.cvtColor(epi, cv2.COLOR_BGR2RGB)
            if i == 0:
                epi = cv2.resize(epi, (512, 9))
            else:
                epi = cv2.resize(epi, (9, 512))

            # Sobel
            x = cv2.Sobel(epi, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(epi, cv2.CV_16S, 0, 1)
            abs_x = cv2.convertScaleAbs(x)
            abs_y = cv2.convertScaleAbs(y)
            dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
            epi = dst

            epi = np.array(epi).astype('float32') / 255
            epi = np.transpose(epi, (2, 0, 1))
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
