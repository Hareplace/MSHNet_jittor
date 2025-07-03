import jittor as jt
import jittor.nn as nn
import jittor
from skimage.measure import label
from skimage import measure


import numpy as np

class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos


    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])


class PD_FA():
    def __init__(self, nclass, bins, size):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.size = size

    def update(self, preds, labels):


        # 确保输入是 Jittor Tensor 并转换为 NumPy
        preds_np = preds.numpy()  # 形状 [4,1,256,256]
        labels_np = labels.numpy()  # 形状 [4,1,256,256]

        for iBin in range(self.bins + 1):
            score_thresh = iBin * (255 / self.bins)

            # 处理 predits（逐个样本处理，避免 batch 维度问题）
            for batch_idx in range(preds_np.shape[0]):  # 遍历 batch 中的每个样本
                # 获取当前样本的预测和标签（去除 channel 维度）
                predits = preds_np[batch_idx, 0, :, :]  # 形状 [256, 256]
                labelss = labels_np[batch_idx, 0, :, :]  # 形状 [256, 256]

                # 二值化并确保数据类型
                predits = (predits > score_thresh).astype('int64')
                labelss = labelss.astype('int64')

                # 检查尺寸是否匹配 self.size
                assert predits.shape == (self.size, self.size), \
                    f"Pred shape {predits.shape} != required {(self.size, self.size)}"
                assert labelss.shape == (self.size, self.size), \
                    f"Label shape {labelss.shape} != required {(self.size, self.size)}"

                # 连通区域分析
                image = measure.label(predits, connectivity=2)
                coord_image = measure.regionprops(image)
                label = measure.label(labelss, connectivity=2)
                coord_label = measure.regionprops(label)

                # 更新指标
                self.target[iBin] += len(coord_label)
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match = []
                self.dismatch = []

                for K in range(len(coord_image)):
                    self.image_area_total.append(coord_image[K].area)

                for i in range(len(coord_label)):
                    centroid_label = np.array(coord_label[i].centroid)
                    for m in range(len(coord_image)):
                        centroid_image = np.array(coord_image[m].centroid)
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        if distance < 3:
                            self.distance_match.append(distance)
                            self.image_area_match.append(coord_image[m].area)
                            del coord_image[m]
                            break

                self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
                self.FA[iBin] += np.sum(self.dismatch)
                self.PD[iBin] += len(self.distance_match)

    def get(self, img_num):

        Final_FA = self.FA / ((self.size * self.size) * img_num)
        Final_PD = self.PD / self.target

        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')
        
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    predict = (jt.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = target.unsqueeze(1).float()  # jittor unsqueeze 替代 np.expand_dims
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum().item()
    fp = (predict * ((predict != target).float())).sum().item()
    tn = ((1 - predict) * ((predict == target).float())).sum().item()
    fn = (((predict != target).float()) * (1 - predict)).sum().item()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp + fp

    return tp, pos, fp, neg, class_pos


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = target.unsqueeze(1).float()   # jittor unsqueeze 替代 np.expand_dims
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum().item()
    pixel_correct = (((predict == target).float() * (target > 0).float())).sum().item()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = target.unsqueeze(1).float()
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    # jittor tensor转numpy
    intersection_np = intersection.numpy().astype(np.int32).flatten()
    predict_np = predict.numpy().astype(np.int32).flatten()
    target_np = target.numpy().astype(np.int32).flatten()

    area_inter, _ = np.histogram(intersection_np, bins=nbins, range=(mini, maxi))
    area_pred, _  = np.histogram(predict_np, bins=nbins, range=(mini, maxi))
    area_lab, _   = np.histogram(target_np, bins=nbins, range=(mini, maxi))
     #jittor 张量需要 .numpy() 转成 numpy 数组，再做 np.histogram

    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
