import torch
import numpy as np


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img_x = sample['d_img_org'][0]
        d_img_y = sample['d_img_org'][1]
        score = sample['score']
        idx = sample['idx']

        d_img_x = (d_img_x - self.mean) / self.var
        d_img_y = (d_img_y - self.mean) / self.var
        sample = {
            'd_img_org': [d_img_x, d_img_y],
            'score': score,
            'idx': idx
        }
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        idx = sample['idx']

        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()

        sample = {
            'd_img_org': d_img,
            'score': score,
            'idx': idx
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img_x = sample['d_img_org'][0]
        d_img_y = sample['d_img_org'][1]
        score = sample['score']
        idx = sample['idx']
        # d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        # score = torch.from_numpy(score).type(torch.FloatTensor)
        # idx   = torch.from_numpy(idx)  .type(torch.FloatTensor)

        d_img_x = torch.from_numpy(d_img_x)
        d_img_y = torch.from_numpy(d_img_y)
        score = torch.from_numpy(score)
        idx = torch.from_numpy(idx)

        # idx = torch.tensor(idx)
        sample = {
            'd_img_org': [d_img_x, d_img_y],
            'score': score,
            'idx': idx
        }
        return sample

