import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from MANIQA.config import config


class IQA_dataset(torch.utils.data.Dataset):
    def __init__(self, config, scene_list, transform, mode='train'):
        super(IQA_dataset, self).__init__()
        self.config = config
        self.scene_list = scene_list  #176 for win5
        self.transform = transform
        self.mode = mode

        self.dis_path = self.config.db_path
        self.txt_file_name = self.config.text_path
        if mode == 'train':
            self.aug_num = self.config.aug_num
        else:
            self.aug_num = 1

        idx_data, dis_files_data, score_data = [], [], []

        #将SAI的水平、垂直、对角分别以3张图片进行堆叠进行输入
        name_list_heng = [['5_1', '5_2', '5_3'], ['5_4', '5_5', '5_6'], ['5_7', '5_8', '5_9']]
        name_list_shu = [['1_5', '2_5', '3_5'], ['4_5', '5_5', '6_5'], ['7_5', '8_5', '9_5']]
        name_list_pie = [['1_1', '2_2', '3_3'], ['4_4', '5_5', '6_6'], ['7_7', '8_8', '9_9']]
        name_list_na = [['9_1', '8_2', '7_3'], ['6_4', '5_5', '4_6'], ['3_7', '2_8', '1_9']]

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

                        # 全取
                        for i in range(len(name_list_sel)):  # 4
                            sai_each = []
                            f_cat = []
                            count = 0

                            for j in range(len(name_list_sel[i])):  # [3,3,3] 确定具体的方向，即水平0、垂直1、左2右3斜对角线

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
