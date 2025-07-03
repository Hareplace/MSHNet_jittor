# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/29 20:48
@Auth ： 孙殿芳
"""
import jittor.nn as nn
import numpy as np
import jittor as jt
from skimage import measure


def SoftIoULoss(pred, target):
    pred = jt.sigmoid(pred)

    smooth = 1

    intersection = pred * target
    intersection_sum = jt.sum(intersection, (1, 2, 3))
    pred_sum = jt.sum(pred, (1, 2, 3))
    target_sum = jt.sum(target, (1, 2, 3))

    loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)

    loss = 1 - loss.mean()

    return loss

def Dice(pred, target, warm_epoch=1, epoch=1, layer=0):
    pred = jt.sigmoid(pred)

    smooth = 1

    intersection = pred * target
    intersection_sum = jt.sum(intersection, (1, 2, 3))
    pred_sum = jt.sum(pred, (1, 2, 3))
    target_sum = jt.sum(target, (1, 2, 3))

    loss = (2 * intersection_sum + smooth) / (pred_sum + target_sum + intersection_sum + smooth)

    loss = 1 - loss.mean()

    return loss


def LLoss(pred, target):
    loss = jt.array(0.0)  # 允许求导

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]

    x_index = jt.arange(0, w).view(1, 1, w).repeat(1, h, 1) / w
    y_index = jt.arange(0, h).view(1, h, 1).repeat(1, 1, w) / h
    smooth = 1e-8

    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_diff = jt.atan(pred_centery / (pred_centerx + smooth)) - jt.atan(target_centery / (target_centerx + smooth))
        angle_loss = (4 / (np.pi ** 2)) * (angle_diff * angle_diff)

        pred_length = jt.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = jt.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = jt.minimum(pred_length, target_length) / (jt.maximum(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss

class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def execute(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = jt.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = jt.sum(intersection, (1, 2, 3))
        pred_sum = jt.sum(pred, (1, 2, 3))
        target_sum = jt.sum(target, (1, 2, 3))

        dis = jt.pow((pred_sum - target_sum) / 2, 2)

        alpha = (jt.minimum(pred_sum, target_sum) + dis + smooth) / (jt.maximum(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)

        lloss = LLoss(pred, target)  # 确认LLoss也迁移到Jittor版本

        if epoch > warm_epoch:
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss

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