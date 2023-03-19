import cv2
import numpy as np
import torch


class cv_transform(object):
    def __init__(self, mean=0.5, std=0.5): #mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        self.mean = mean
        self.std = std

    def __call__(self, sample):   #[3, 512, 512]
        sample = np.array(sample).astype('float32') / 255
        sample = (sample - self.mean) / self.std
        # sample = np.transpose(sample, (2, 0, 1))   modify
        sample = torch.from_numpy(sample).type(torch.FloatTensor)

        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        sai = sample
        sai = (sai - self.mean) / self.var

        return sai


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sai = sample

        # d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        # score = torch.from_numpy(score).type(torch.FloatTensor)
        # idx   = torch.from_numpy(idx)  .type(torch.FloatTensor)

        sai = torch.from_numpy(sai)

        return sai


# def sobel(sample):
#     x = cv2.Sobel(sample, cv2.CV_16S, 1, 0)
#     y = cv2.Sobel(sample, cv2.CV_16S, 0, 1)
#     abs_x = cv2.convertScaleAbs(x)
#     abs_y = cv2.convertScaleAbs(y)
#     dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
#
#     return dst
#
#
# def sobel_gray(sample):
#
#     x = cv2.Sobel(sample, cv2.CV_64F, 1, 0, ksize=3)
#     y = cv2.Sobel(sample, cv2.CV_64F, 0, 1, ksize=3)
#     dst = cv2.addWeighted(x, 0.5, y, 0.5, 0)
#
#     return dst
