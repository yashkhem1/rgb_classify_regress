import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
from numpy.random import randn
import cv2


def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    print('New Learning Rate: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_accuracy(desp_a, desp_p, desp_n, margin):
    dist_ap = np.linalg.norm(desp_a-desp_p, axis=1, keepdims=True)
    dist_an = np.linalg.norm(desp_a-desp_n, axis=1, keepdims=True)
    dist_diff = dist_an - dist_ap
    #print('ap'+str(dist_ap)+' an '+ str(dist_an)+ ' dist_dif '+ str(dist_diff))
    acc = np.average(dist_diff > 0.6 * margin)

    return acc


def Rnd(x):
	return max(-2 * x, min(2 * x, randn() * x))


def Flip(img):
	return img[:, :, ::-1].copy()


# def ShuffleLR(x):
# 	for e in ref.shuffleRef:
# 		x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
# 	return x