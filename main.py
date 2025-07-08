from jittor.optim import Adam
from utils.data import *               
from utils.metric import *            
from argparse import ArgumentParser

import jittor as jt                 
import jittor.dataset as dataset  
from model.MSHNet import *            
from model.loss import *              

from tqdm import tqdm
import os
import os.path as osp
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  




def parse_args():
    parser = ArgumentParser(description='Implement of model (Jittor version)')

    # 数据集与训练设置
    parser.add_argument('--dataset-dir', type=str, default='dataset/IRSTD1k')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    # 输入图像处理尺寸
    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)

   
    parser.add_argument('--multi-gpus', type=bool, default=False)         # 可保留，但 Jittor 自动管理 GPU
    parser.add_argument('--if-checkpoint', type=bool, default=False)      # 是否加载 checkpoint
    parser.add_argument('--mode', type=str, default='train')              # train / test
    parser.add_argument('--weight-path', type=str, default='weights/iou_67.86_IRSTD-1k_jittor.npz')  # 改为适合 Jittor 的文件名格式

    args = parser.parse_args()
    return args


class Trainer(object):

    def visualize_prediction(self, image, pred, gt, save_path):
        """
        显示：原图 + 预测掩码（黑白+绿框） + 真实掩码（黑白+红框）
        """
        import cv2
        import numpy as np

        # ---- Step 1: Tensor -> Numpy ----
        image = image.numpy()
        pred = pred.numpy()
        gt = gt.numpy()

        # ---- Step 2: 原图恢复 ----
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # [C,H,W] -> [H,W,C]
        else:
            image = np.stack([image.squeeze()] * 3, axis=-1)  # 灰度图重复 3 通道
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # ---- Step 3: Mask 预处理为二值图 ----
        pred_mask = (pred > 0.5).astype(np.uint8).squeeze() * 255  # shape: [H,W]
        gt_mask = gt.squeeze().astype(np.uint8) * 255              # shape: [H,W]

        # ---- Step 4: 将二值图转为3通道黑白图，并加轮廓框 ----
        def make_mask_vis(mask, color=(0,255,0)):
            vis = np.stack([mask]*3, axis=-1)  # gray -> 3 channels
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
            return vis

        pred_vis = make_mask_vis(pred_mask, color=(0,255,0))  # green for pred
        gt_vis   = make_mask_vis(gt_mask,   color=(0,0,255))  # red for GT

        # ---- Step 5: 拼接三图 ----
        concat = cv2.hconcat([image, pred_vis, gt_vis])
        cv2.imwrite(save_path, concat)

    def __init__(self, args):
        assert args.mode in ['train', 'test']

        self.args = args
        self.start_epoch = 0
        self.mode = args.mode

        # 数据集
        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        self.train_loader = trainset
        self.val_loader = valset  # 验证集同理
        # 设置设备（Jittor 默认自动选择 GPU，如可用）
        jt.flags.use_cuda = jt.has_cuda  # 或手动指定：jt.flags.use_cuda = True / False
        self.device = "cuda" if jt.flags.use_cuda else "cpu"

        # 初始化模型
        self.model = MSHNet(3)

        # Jittor 会自动使用多个 CUDA 核心，无需手动调用 DataParallel
        if args.multi_gpus and jt.has_cuda:
            print(f"Using Jittor with CUDA, device count: {jt.cuda.device_count()}")

        # 优化器，Jittor的Adagrad不需要filter，直接传参数即可
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)

        # 最大池化层，Jittor用 nn.MaxPool2d，参数一样
        self.down = nn.MaxPool2d(2, 2)

        
        self.loss_fun = SLSIoULoss()

        
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC = ROCMetric(1, 10)

        # 记录最优IoU和warm epoch参数
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch
        
        # 记录训练log
        self.timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.result_dir = 'result'
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.result_csv_path = osp.join(self.result_dir, f'train_log_{self.timestamp}.csv')

        # 权重加载
        if args.mode == 'train':
            if args.if_checkpoint and osp.exists(args.weight_path):
                print(f"Loading checkpoint from: {args.weight_path}")
                self.model.load(args.weight_path)  # 直接加载 .npz 权重文件

                # 如果未来有保存 optimizer 的权重，可以用 jt.load 加载 dict
                # checkpoint = jt.load(args.weight_path)
                # self.model.load_parameters(checkpoint['model'])
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                # self.start_epoch = checkpoint.get('epoch', 0) + 1
                # self.best_iou = checkpoint.get('iou', 0.0)

                self.save_folder = os.path.dirname(args.weight_path)
            else:
                # 没有 checkpoint，创建新的保存路径
                self.save_folder = f'weights/MSHNet-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
                if not osp.exists(self.save_folder):
                    os.makedirs(self.save_folder)

                # ✅ [新增] 初始化 loss 日志文件路径
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                self.loss_log_file = osp.join(self.save_folder, f"loss_{timestamp}.txt")
                self.all_epoch_losses = []  # 存放每轮的平均 loss


        if args.mode == 'test':
            if args.weight_path and osp.exists(args.weight_path):
                print(f"Loading Jittor weights from: {args.weight_path}")
                self.model.load(args.weight_path)  # 自动加载结构和参数

            else:
                print("Warning: No valid weight_path provided. Using randomly initialized model.")
            self.warm_epoch = -1

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = False
        
        # 性能记录开始时间
        start_time = time.time()
        total_batch_time = 0

        for i, (data, mask) in enumerate(tbar):
            batch_start_time = time.time()  # 记录 batch 开始时间

            # 无需 .to(self.device)，Jittor 自动管理设备
            if epoch > self.warm_epoch:
                tag = True

            masks, pred = self.model(data, tag)
            loss = self.loss_fun(pred, mask, self.warm_epoch, epoch)

            for j in range(len(masks)):
                if j > 0:
                    mask = nn.Pool(2, 2, op="maximum")(mask)  # Jittor 中的 MaxPool2d
                loss += self.loss_fun(masks[j], mask, self.warm_epoch, epoch)

            loss = loss / (len(masks) + 1)

            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()

            losses.update(loss.item(), pred.shape[0])
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
            
            batch_end_time = time.time()  # 记录 batch 结束时间
            total_batch_time += batch_end_time - batch_start_time  # 累加 batch 时间

            # ✅ [新增] 记录当前 epoch 的 loss 值
            self.all_epoch_losses.append(losses.avg)

            with open(self.loss_log_file, "a") as f:
                f.write(f"{epoch + 1},{losses.avg:.6f}\n")
                
        end_time = time.time()
        epoch_time = end_time - start_time
        if len(self.train_loader) > 0:
            avg_batch_time = total_batch_time / len(self.train_loader)
            fps = self.args.batch_size / avg_batch_time
        else:
            avg_batch_time = 0
            fps = 0

        # 写入性能日志（ txt ）
        perf_log_path = osp.join(self.result_dir, f'train_perf_log_{self.timestamp}.txt')
        with open(perf_log_path, 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Epoch time: {epoch_time:.2f} s\n")
            f.write(f"  Avg batch time: {avg_batch_time * 1000:.2f} ms\n")
            f.write(f"  Train FPS: {fps:.2f} samples/sec\n\n")
        
        
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader)
        tag = False

        with jt.no_grad():
            for i, (data, mask) in enumerate(tbar):
                data = data  # Jittor默认会自动放到设备，无需to(device)
                mask = mask

                if epoch > self.warm_epoch:
                    tag = True

                _, pred = self.model(data, tag)  # pred是Jittor张量

                # 1. mIoU用Jittor张量直接传
                self.mIoU.update(pred, mask)

                # 2. ROCMetric用Jittor张量直接传
                self.ROC.update(pred, mask)


                self.PD_FA.update(pred, mask)

                # 计算当前mIoU，用于显示
                _, mean_IoU = self.mIoU.get()
                tbar.set_description(f'Epoch {epoch}, IoU {mean_IoU:.4f}')

        # 计算最终指标
        FA, PD = self.PD_FA.get(len(self.val_loader))
        _, mean_IoU = self.mIoU.get()
        tp_rate, fp_rate, recall, precision = self.ROC.get()

        print(f"Final Results - IoU: {mean_IoU:.4f}, FA: {FA}, PD: {PD}")

        if self.mode == 'train':
             # 训练指标写入以时间戳命名的 CSV 文件
                if not osp.exists(self.result_csv_path):
                    with open(self.result_csv_path, 'w') as f:
                        f.write("epoch,IoU,PD,FA\n")
                with open(self.result_csv_path, 'a') as f:
                    f.write(f"{epoch},{mean_IoU:.4f},{PD[0]:.4f},{FA[0] * 1e6:.4f}\n")
            
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU

                    # 保存最优权重
                    jt.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))

                    # 写入日志
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                            epoch, self.best_iou, PD[0], FA[0] * 1000000))

                # 保存 checkpoint（包含优化器状态）
                checkpoint = {
                    'net': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'iou': self.best_iou
                }
                jt.save(checkpoint, osp.join(self.save_folder, 'checkpoint.pkl'))

        elif self.mode == 'test':
            print('mIoU: {:.4f}\n'.format(mean_IoU))
            print('Pd: {:.4f}\n'.format(PD[0]))
            print('Fa: {:.4f}\n'.format(FA[0] * 1000000))


        # ✅ 保存可视化图像（仅训练最后几轮）
        if self.mode == 'train' and epoch >= self.args.epochs - 5:  # 最后5轮保存
            save_vis_dir = osp.join(self.result_dir, 'vis', f'epoch_{epoch}')
            os.makedirs(save_vis_dir, exist_ok=True)

            with jt.no_grad():
                for i, (data, mask) in enumerate(self.val_loader):
                    if i >= 5:
                        break  # 每轮只保存前5张
                    _, pred = self.model(data, tag)
                    for b in range(data.shape[0]):
                        img = data[b]
                        pd = pred[b]
                        gt = mask[b]

                        # save_path = osp.join(save_vis_dir, f"sample_{i*data.shape[0]+b}.png")
                        save_path = osp.join(save_vis_dir, f"img_{i * self.args.batch_size + b}.png")

                        self.visualize_prediction(img, pd, gt, save_path,)



if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)  

    if trainer.mode == 'train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)

        # ✅ [新增] 训练结束提示日志路径
        print(f"\n✅ Loss 日志已保存到：{trainer.loss_log_file}")
    else:
        trainer.test(1)
