import os
import random
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

from MANIQA.config import config


class IQA_dataset(torch.utils.data.Dataset):
    def __init__(self, config, scene_list, transform, mode='train'):
        super(IQA_dataset, self).__init__()
        self.config = config
        self.scene_list = scene_list
        self.transform = transform
        self.mode = mode

        self.dis_path = self.config.db_path
        self.txt_file_name = self.config.text_path
        self.aug_num = self.config.aug_num

        idx_data, dis_files_data, score_data = [], [], []

        name_list_heng = [['004_000', '004_001', '004_002'], ['004_003', '004_004', '004_005'], ['004_006', '004_007', '004_008']]
        name_list_shu  = [['000_004', '001_004', '002_004'], ['003_004', '004_004', '005_004'], ['006_004', '007_004', '008_004']]
        name_list_pie  = [['000_000', '001_001', '002_002'], ['003_003', '004_004', '005_005'], ['006_006', '007_007', '008_008']]
        name_list_na   = [['008_000', '007_001', '006_002'], ['005_003', '004_004', '003_005'], ['002_006', '001_007', '000_008']]

        name_list_sel = [
            name_list_heng,
            name_list_shu,
            name_list_pie,
            name_list_na
        ]

        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                idx, dis, score = line.split()
                idx = int(idx)
                score = float(score)

                if idx in self.scene_list:
                    for aug_num in range(self.aug_num):
                        # f_cat = []

                        # 全取
                        for i in range(len(name_list_sel)):  # 4
                            sai_each = []
                            f_cat = []
                            count = 0

                            for j in range(len(name_list_sel[i])):  # [3,3,3]

                                for n in range(len(name_list_sel[i][j])):
                                    each = '{}/{}.png'.format(dis, name_list_sel[i][j][n])
                                    sai_each.append(each)
                                    count += 1
                                if count >= len(name_list_sel[i][j]):  # 3
                                    f_cat.append(sai_each)
                                    sai_each = []
                                    count = 0
                            dis_files_data.append(f_cat)
                            idx_data.append(idx)
                            score_data.append(score)
                            # print(idx, '\t', f_cat, '\t', score)
                            # exit()
        # exit()
        # reshape list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        idx_data = np.array(idx_data)
        idx_data = idx_data.reshape(-1, 1)

        self.data_dict = {
            'd_img_list': dis_files_data,
            'score_list': score_data,
            'idx_list': idx_data
        }

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        h, w = self.config.input_size
        top = random.randint(0, h - config.crop_size)
        left = random.randint(0, w - config.crop_size)
        bottom = top + config.crop_size
        right = left + config.crop_size

        if_flip = random.random()  # 随机数

        cat_all = []
        '''1-'''
        for n in range(len(self.data_dict['d_img_list'][idx])):
            dis = []
            for i in range(len(self.data_dict['d_img_list'][idx][n])):
                d_img_name = self.data_dict['d_img_list'][idx][n][i]
                d_img = Image.open(Path(self.config.db_path) / d_img_name).convert("RGB")

                d_img = d_img.resize(self.config.input_size)

                '''翻转、裁剪'''
                if if_flip > 0.5 and self.mode == 'train':
                    d_img = d_img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转, ++

                d_img = d_img.crop((left, top, right, bottom))

                if self.transform:
                    d_img = self.transform(d_img)
                dis.append(d_img)

            dis = torch.cat(dis, dim=0)

            cat_all.append(dis)

        score = self.data_dict['score_list'][idx]
        idx = self.data_dict['idx_list'][idx]
        sample = {
            'd_img_org': cat_all,
            'score': score,
            'idx': idx
        }

        return sample

