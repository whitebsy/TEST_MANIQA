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

        name_list_1 = [['000', '001', '002'], ['003', '004', '005'], ['006', '007', '008']]
        # name_list_2 = [['046', '047', '048'], ['049', '050', '051'], ['052', '053', '054']]
        # name_list_3 = [['092', '093', '094'], ['095', '096', '097'], ['098', '099', '100']]
        name_list_3 = [['100', '099', '098'], ['097', '096', '095'], ['094', '093', '092']]

        # name_list_sel = [name_list_1, name_list_2, name_list_3]
        name_list_sel = [name_list_1, name_list_3]

        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                idx, dis, score = line.split()
                idx = int(idx)
                score = float(score)

                if idx in self.scene_list:
                    for aug_num in range(self.aug_num):

                        # 全取
                        for i in range(len(name_list_sel)):  # 4
                            sai_each = []
                            f_cat = []
                            count = 0

                            for j in range(len(name_list_sel[i])):  # [3,3,3]

                                for n in range(len(name_list_sel[i][j])):
                                    each = '{}/Frame_{}.png'.format(dis, name_list_sel[i][j][n])
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
        h, w = 720, 960
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

                # d_img = d_img.resize(self.config.input_size)

                '''翻转、裁剪'''
                if if_flip > 0.5 and self.mode == 'train':
                    d_img = d_img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转, ++

                d_img = d_img.crop((left, top, right, bottom))

                if self.transform:
                    d_img = self.transform(d_img)
                dis.append(d_img)

            dis = torch.cat(dis, dim=0)

            cat_all.append(dis)
        # cat_all = np.array(cat_all)
        # cat_all = torch.from_numpy(cat_all)

        score = self.data_dict['score_list'][idx]
        idx = self.data_dict['idx_list'][idx]
        sample = {
            'd_img_org': cat_all,
            'score': score,
            'idx': idx
        }

        return sample

