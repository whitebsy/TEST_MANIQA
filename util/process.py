import torch
import numpy as np


def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config.crop_size)
    left = np.random.randint(0, w - config.crop_size)
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        idx   = sample['idx']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        ret_d_img = d_img[:, top: top + new_h, left: left + new_w]

        sample = {
            'd_img_org': ret_d_img,
            'score': score,
            'idx': idx
        }
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        idx   = sample['idx']

        d_img = (d_img - self.mean) / self.var
        sample = {
            'd_img_org': d_img,
            'score': score,
            'idx': idx
        }
        return sample


class NM_Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        idx   = sample['idx']
        score = sample['score']

        d_img_org[:, :, 0] = (d_img_org[:, :, 0] - self.mean[0]) / self.var[0]
        d_img_org[:, :, 1] = (d_img_org[:, :, 1] - self.mean[1]) / self.var[1]
        d_img_org[:, :, 2] = (d_img_org[:, :, 2] - self.mean[2]) / self.var[2]

        sample = {'d_img_org': d_img_org, 'score': score, 'idx': idx}

        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample['d_img_org']
        score = sample['score']
        idx   = sample['idx']

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
        d_img = sample['d_img_org']
        score = sample['score']
        idx   = sample['idx']
        # d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        # score = torch.from_numpy(score).type(torch.FloatTensor)
        # idx   = torch.from_numpy(idx)  .type(torch.FloatTensor)

        d_img = torch.from_numpy(d_img)
        score = torch.from_numpy(score)
        idx   = torch.from_numpy(idx)

        # idx = torch.tensor(idx)
        sample = {
            'd_img_org': d_img,
            'score': score,
            'idx': idx
        }
        return sample

