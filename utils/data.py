import jittor as jt
from jittor import nn
import jittor.dataset as dataset 
import numpy as np
import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import shutil


class IRSTD_Dataset(dataset.Dataset):
    def __init__(self, args, mode='train'):
        super().__init__()
        self.dataset_dir = args.dataset_dir

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'
        else:
            raise ValueError("Unknown mode")

        self.list_dir = osp.join(self.dataset_dir, txtfile)
        self.imgs_dir = osp.join(self.dataset_dir, 'images')
        self.label_dir = osp.join(self.dataset_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size

        self.mean = [.485, .456, .406]
        self.std = [.229, .224, .225]

        self.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=4)  

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unknown mode")

        img = self.to_tensor(img)
        img = self.normalize(img, self.mean, self.std)

        mask = self.to_tensor(mask)

        return img, mask

    def __len__(self):
        return len(self.names)

    def to_tensor(self, pic):
        # 将 PIL 图片转换为 jittor tensor，C×H×W，float32归一化
        img = np.array(pic).astype('float32') / 255.0
        if img.ndim == 2:  # 灰度
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)
        return jt.array(img)

    def normalize(self, tensor, mean, std):
        # tensor shape C×H×W
        # 这里用循环给每个通道做归一化
        res = []
        for t, m, s in zip(tensor, mean, std):
            t = (t - m) / s
            res.append(t)
        return jt.stack(res, dim=0)

    
   
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        return img, mask
