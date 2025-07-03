>下文中数据准备方式、训练脚本和测试脚本的相关命令均适用于MSHNet-pytorch版本。
>
>pytorch版本所有源码、结果对齐部分提到的pytorch版本的所有图像可在笔者上传的MSHNet_pytorch仓库中查看：https://github.com/Hareplace/MSHNet_pytorch

# 环境配置

#### 1. 本地开发环境（Windows）

- **操作系统**: Windows 10

- **Python版本**: 3.8

- **训练硬件**: CPU(12th Gen Intel(R) Core(TM) i5-1240P)

- **关键依赖项**:

  - **Jittor**: 1.3.8.1
  - **scikit-image**: 最新版（图像处理库，需`measure.label`和`regionprops`功能）
  - **pandas**:1.5.0（读取log.csc，绘图需要)

- **安装命令**:

  ```
  # 安装Jittor（指定清华镜像加速）
  pip install jittor==1.3.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  
  # 安装scikit-image（必须最新版以确保功能兼容性）
  pip install --upgrade scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
  
  #安装pandas
  pip install pandas==1.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

#### 2. 远程GPU环境（AutoDL）

- **操作系统**: Ubuntu 18.04

- **Python版本**: 3.8

- **训练硬件**: NVIDIA RTX 3090 (24GB)

- **CUDA版本**: 11.3（需与Jittor版本匹配）

- **关键依赖项**:

  - **Jittor**: 1.3.1（AutoDL镜像预装版本，**不建议升级**，避免g++编译器兼容性问题）
  - **scikit-image**: 最新版（与本地环境一致）
  - **pandas**：1.5.0（与本地环境一致）

- **安装命令**:

  ```
  # 安装scikit-image（Jittor已预装）
  pip install --upgrade scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
  
  #安装pandas
  pip install pandas==1.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

------

### **关键依赖项说明**

| 库/模块                | 用途                                                         | 版本要求                          |
| :--------------------- | :----------------------------------------------------------- | :-------------------------------- |
| `jittor`               | 深度学习框架，支持自动微分和GPU加速                          | 本地：1.3.8.1 远程：1.3.1（固定） |
| `scikit-image.measure` | 提供图像分析工具： - `label`: 连通区域标记 - `regionprops`: 区域属性统计 | 最新版                            |

------

### **注意事项**

1. **版本兼容性**:
   - AutoDL环境中预置的Jittor 1.3.1与CUDA 11.3和系统g++编译器绑定，手动升级可能导致编译错误。
   - 本地开发若需GPU支持，需额外配置Jittor的CUDA环境变量（参考[官方文档](https://cg.cs.tsinghua.edu.cn/jittor/)）。
2. **镜像加速**:
   - 国内用户建议使用清华镜像源（`-i https://pypi.tuna.tsinghua.edu.cn/simple`）加速安装。

---

# **数据准备**

本实验选用 **IRSTD-1k** 数据集进行训练与评估，原因如下：

1. **模型适配性**：
   原论文作者在 **IRSTD-1k** 上训练并验证了模型的最佳性能（如 IoU 67.86），该数据集经过专门优化，适用于红外小目标检测任务，能充分验证模型的泛化能力。
2. **数据特性优势**：
   - **高质量标注**：提供像素级精确标注，背景与目标区分明显，适合监督学习。
   - **场景多样性**：涵盖不同复杂背景（如天空、地面、海面等），增强模型鲁棒性。
   - **标准评测基准**：官方划分的 `trainval.txt` 和 `test.txt` 可确保实验对比的公平性。
3. **即用性**：
   数据集已预处理为可直接训练的格式，仅需调整文件夹命名即可匹配代码（如 `IRSTD1k_Img/` → `images/`），无需额外数据清洗或增强。

#### 1. 数据集来源（IRSTD1k）

- **官方地址**：
- [GitHub - RuiZhang97/ISNet: CVPR2022 ''ISNet: Shape Matters for Infrared Small Target Detection''](https://github.com/RuiZhang97/ISNet)
  - 推荐通过Google Drive下载链接获取完整数据。
  - 备用下载方式（百度网盘）。

#### 2. 数据集目录结构

下载后解压的文件夹应包含以下内容：

```
IRSTD-1k/
├── IRSTD1k_Img/                 
│   ├── XDU0.png					# 原始图像（.png格式）
│   └── ...
├── IRSTD1k_Label/                # 对应的二值标注掩码（前景为白色，背景为黑色）
│   ├── XDU0.png
│   └── ...
├── test.txt              # 测试集文件名列表（无后缀）
├── trainval.txt          # 训练+验证集文件名列表
└── trainvaltest.txt      # 全集文件名列表
```

#### 3.快速配置步骤

在MSHNet_jittor总目录下，添加**dataset**目录，将IRSTD1k文件夹复制粘贴进去即可，但要注意**将原始文件夹重命名**，使其匹配代码中的默认路径约定：

```
IRSTD-1k/                → 改为 IRSTD1k/
  ├── IRSTD1k_Img/       → 改为 images/
  └── IRSTD1k_Label/     → 改为 masks/
  └──test.txt	         → 不变，直接复制即可
  └──trainval.txt        → 不变，直接复制即可
  └──trainvaltest.txt    → 不变，直接复制即可
```

**注意**：本仓库以及pytorch版本仓库中不包括数据集文件，需自行下载然后按照上述操作配置。

### **注意事项**

1. **无需预处理**：
   标注已是标准的二值图（0为背景，255为前景），可直接用于训练
2. **数据集划分**：
   直接使用官方提供的txt文件（`test.txt`/`trainval.txt`）作为数据分割依据

# 训练脚本

#### 1. 脚本启动方式

```
python main --dataset-dir --batch-size --epochs --lr --mode "train"
```
如：
```
python main --dataset-dir "dataset/IRSTD1k" --batch-size 4 --epochs 100 --lr 0.01 --mode "train"
```
#### 2. 核心参数配置

在命令行参数中应包含：

```
# 数据相关
--dataset-dir  "dataset/IRSTD1k"  # 数据集根目录，注意要用双引号，单引号会无法识别参数
# 训练超参数
--batch-size 4
--epochs 100
--lr 0.01
#以上涉及参数为得到下文中结果与日志的命令行参数设置（即笔者实际运行的训练命令），训练命令还可添加以下参数：
--warm-epoch   热身训练轮数，前N轮仅训练部分网络层（如解码器）
--base-size    图像输入的基础尺寸
--crop-size    随机裁剪的尺寸
--multi-gpus   是否使用多GPU（Jittor自动管理，实际无需手动设置）
--if-checkpoint   是否从检查点恢复训练（需配合--weight-path使用）
--weight-path  权重路径
```

#### 3. 关键代码结构

```
# main.py 
    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = False

        for i, (data, mask) in enumerate(tbar):
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
```

# 测试脚本

#### 1. 脚本启动方式

```
python main.py --dataset-dir "dataset/IRSTD1k" --batch-size 4 --mode "test" --weight-path "/weights/iou_67.86_IRSTD-1k_jittor.npz"
```

**权重转换说明**：原始PyTorch权重(`.pkl`)需通过`convert_weights_to_jittor.py`转换为Jittor格式(`.npz`)，验证是否完全转换成功可使用`test.py`，在源码中均给出。

#### 2. 核心参数配置

在命令行参数中应包含：

```
# 数据相关
--dataset-dir  "dataset/IRSTD1k"  # 数据集根目录，注意要用双引号，单引号会无法识别参数
# 测试超参数
--batch-size 4
--weight-path "weights/iou_67.86_IRSTD-1k_jittor.npz"
```

#### 3. 关键代码结构

```
# main.py 
        def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader)
        tag = False

        with jt.no_grad():
            for i, (data, mask) in enumerate(tbar):
                data = data 
                mask = mask

                if epoch > self.warm_epoch:
                    tag = True

                _, pred = self.model(data, tag)  # pred是Jittor张量

                self.mIoU.update(pred, mask)
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
```

# 与 PyTorch 结果对齐

##  训练日志

### 训练log

两个版本的训练log均保存在`result`目录下，以`csv`形式存储，形如`train_log_时间戳.csv`，记录了随着训练的进行，IoU、PD、FA指标的情况，可自行查看,如https://github.com/Hareplace/MSHNet_jittor/blob/master/result/train_log_2025-07-02-20-47-09.csv

在`utils`文件夹下有`plot_log.py`，可将该`csv`可视化，运行命令示例为：

```
python utils/plot_log.py "result/train_log_2025-07-02-09-11-22.csv"
```

可视化后的图像将保存在`result`目录下的`log_curve`中（若不存在则可自动创建）。

两个版本的train_log图像如下所示：

#### jittor版：

<img src="https://github.com/Hareplace/MSHNet_jittor/blob/master/result/log_curve/train_log_2025-07-02-20-47-09_metrics.png"  />

- **IoU（蓝色）**：随着训练轮次的增加，IoU值呈现出上升趋势，尽管存在一些波动，但总体趋势是上升的。
- **PD（绿色）**：PD值从较高水平迅速下降，并在较低水平上趋于稳定。
- **FA（红色）**：FA值在初期迅速下降，然后保持在一个相对稳定的水平，表明随着训练的进行，误报率逐渐降低并趋于稳定。

#### pytorch版：

<img src="https://github.com/Hareplace/MSHNet_pytorch/blob/master/result/log_curve/train_log_2025-07-02-09-11-22_metrics.png" />

- **IoU（蓝色）**：IoU值同样随着训练轮次的增加而上升，且波动性较小，显示出较为稳定的上升趋势。
- **PD（绿色）**：PD值同样从较高水平迅速下降，并在较低水平上趋于稳定，与Jittor版本相似。
- **FA（红色）**：FA值在初期迅速下降，然后保持在一个相对稳定的水平，与Jittor版本相似。

#### 推论

Jittor版本和PyTorch版本在训练过程中的指标表现相似，主要区别在于**IoU曲线的波动性**。PyTorch版本在IoU曲线上显示出更稳定的上升趋势，而Jittor版本则表现出更大的波动性。两个版本在PD和FA曲线上的表现比较相似。

### loss曲线

两个版本训练过程中的loss日志保存在`weights`下每次训练得到命名为`MSHNet-时间戳`的文件夹中，以`txt`形式存储，形如`loss_时间戳.txt`，记录了随着训练的进行，loss值变化的情况，可自行查看，如https://github.com/Hareplace/MSHNet_jittor/blob/master/weights/MSHNet-2025-07-01-16-37-59/loss_20250701-163759.txt

在`utils`文件夹下有`plot_loss.py`，可将该`txt`可视化，运行命令示例为：

```
python utils/plot_loss.py "weights/MSHNet-2025-07-01-23-30-50/loss_20250701-233050.txt"
```

可视化后的图像将保存在`weights`目录下的`loss_curve`中（若不存在则可自动创建）。

两个版本的loss曲线如下所示：

#### jittor版：

<img src="https://github.com/Hareplace/MSHNet_jittor/blob/master/weights/loss_curve/loss_20250701-163759.png" />

#### Jittor 版本 Loss 曲线分析

- **初始 loss ≈ 0.9 左右**，很快出现一次尖锐上升至 **≈ 1.65**（epoch ≈ 3）。
- 随后快速下降，**在 epoch ≈ 10 降至 1.0 以下**。
- **整体曲线呈现收敛趋势**，loss 在 20 ~ 100 epoch 间波动较小，大致稳定在 **0.78 ~ 0.85** 区间。
- 说明模型很早进入稳定训练阶段，且无明显的震荡或发散现象。

**特点**：

- 收敛速度快，约10个epoch以内进入稳定区间。
- 中后期loss曲线平稳，训练过程稳定性好。
- 早期有一次尖峰（可能是学习率设定导致），但影响不大。

#### pytorch版：

<img src="https://github.com/Hareplace/MSHNet_pytorch/blob/master/weights/loss_curve/loss_20250702-091122.png" />

####   PyTorch 版本 Loss 曲线分析

- **初始 loss 也为 ≈ 0.9 左右**，同样在 early epoch（≈2）出现高峰 **≈ 1.75**，高于 Jittor。
- **下降过程较缓慢**，尤其在 10 ~ 40 epoch 阶段，曲线抖动较明显。
- **直到 epoch ≈ 60 后，loss 才逐步收敛至 0.8 以下**，并继续缓慢下降。
- **训练过程抖动幅度较大**，表现出训练波动性更强。

 **特点**：

- 收敛速度慢（至少40~60 epoch 才趋于收敛）。
- loss下降更缓慢，说明优化路径更“犹豫”。
- 震荡幅度较大，训练稳定性不如 Jittor。

| 维度          | Jittor 版            | PyTorch 版             | 分析               |
| ------------- | -------------------- | ---------------------- | ------------------ |
| 初始 Loss     | ≈ 0.9                | ≈ 0.9                  | 持平               |
| 最大波峰      | **1.65**（早期尖峰） | **1.75**（更剧烈）     | Jittor 更温和      |
| 收敛速度      | 快，10 epoch 内趋稳  | 慢，约60 epoch         | Jittor 优          |
| 最终稳定 Loss | ≈ 0.78~0.85          | ≈ 0.75（但过程更波动） | PyTorch 略低但不稳 |
| 震荡幅度      | 小，收敛曲线平滑     | 大，有明显震荡         | Jittor 更稳        |

#### 推论：

 从两组训练 loss 曲线可以看出，Jittor 版本在训练初期虽然出现一次小幅震荡，但整体下降速度更快，约在 10 个 epoch 内就进入稳定阶段，后续 loss 曲线平滑，训练稳定性良好。而 PyTorch 版本的 loss 曲线在前 50 个 epoch 表现出较大的震荡，收敛速度较慢，最终 loss 虽略低于 Jittor，但训练过程明显不如后者平稳。

 因此，从 **训练稳定性与收敛效率** 的角度看，Jittor 更具工程实用价值，适合快速部署场景；而 PyTorch 虽在最终 loss 略占优势，但需更长时间调整优化路径。

## 性能日志

两个版本的训练性能log均保存在`result`目录下，以`txt`形式存储，形如`train_pref_log_时间戳.txt`，记录每轮训练花费时间（ Epoch time）batch平均花费时间（ Avg batch time）以及Train FPS的情况，可自行查看，如https://github.com/Hareplace/MSHNet_jittor/blob/master/result/train_perf_log_2025-07-02-20-47-09.txt

在`utils`文件夹下有`plot_log.py`，可将该`txt`可视化，运行命令示例为：

```
python utils/plot_log.py "result/train_perf_log_2025-07-02-09-11-22.txt"
```

可视化后的图像将保存在`result`目录下的`log_curve`中（若不存在则可自动创建）。

两个版本的train_pref_log图像如下所示：

#### jittor版：

<img src="https://github.com/Hareplace/MSHNet_jittor/blob/master/result/log_curve/train_perf_log_2025-07-02-20-47-09_performance.png" />

#### Jittor版本性能log图像分析：

- **Epoch Time（蓝色）**：在训练初期波动较大，随后趋于稳定，在某些epoch（如第40和第90个epoch）出现显著的峰值，但平均时间**约为14s**
- **FPS（红色）**：在训练初期波动较大，随后趋于稳定，整体保持在**240samples/sec**左右。
- **Batch Time（绿色）**：在训练初期波动较大，随后趋于稳定，整体保持在**12毫秒**左右。

#### pytorch版：

<img src="https://github.com/Hareplace/MSHNet_pytorch/blob/master/result/log_curve/train_perf_log_2025-07-02-09-11-22_performance.png" />

#### Pytorch版本性能log图像分析：

- **Epoch Time（蓝色）**：整体趋势是先下降后上升，波动幅度一直较大，平均时间约为**35s**。

- **FPS（红色）**：波动较小，整体保持在**30samples/sec**左右。

- **Batch Time（绿色）**：波动较大，整体保持在**30-40毫秒**之间。

以下是Jittor版本和PyTorch版本的性能对比分析表格：

| 性能指标       | Jittor版本                    | PyTorch版本              | 对比分析                       |
| :------------- | :---------------------------- | :----------------------- | :----------------------------- |
| **Epoch Time** | 平均约14s，初期波动后趋于稳定 | 平均约35s，波动幅度较大  | Jittor比PyTorch快约2.5倍       |
| **FPS**        | 稳定在240 samples/sec左右     | 稳定在30 samples/sec左右 | Jittor的FPS是PyTorch的8倍      |
| **Batch Time** | 稳定在12毫秒左右              | 波动在30-40毫秒之间      | Jittor的Batch Time更低且更稳定 |

#### 推论

Jittor版在训练过程中表现出更高的效率和稳定性，**Epoch Time**：Jittor版平均时间较短，处理每个epoch的效率更高；**FPS**：Jittor版每秒处理的样本数更多，处理样本的效率更高；**Batch Time**：Jittor版处理每个batch的时间更短，处理batch的效率更高。

Jittor在Epoch Time、FPS和Batch Time上均显著优于PyTorch，尤其是在FPS方面表现突出，且Jittor的训练过程更加稳定，波动较小，而PyTorch的波动幅度较大，Jittor的整体训练效率更高，适合需要快速迭代的场景。

相比之下，PyTorch版在处理每个epoch、样本和batch时需要更多的时间，效率相对较低。这应该就是与Jittor版在底层优化和并行计算方面的优势有关。

## 结果

已知`metric.log`包含**性能提升时记录**：只有当在验证集上计算出的平均IoU (`mean_IoU`) 超过了之前记录的最佳IoU (`self.best_iou`) 时，才会将当前epoch的信息写入`metric.log`文件，可以监控模型性能的提升，并记录下性能最好的模型状态。

`metric.log`记录的是模型在训练过程中性能最优的几个特定轮次，以下指标为笔者在`metric.log`中选择的综合最佳表现指标。

### 关键指标对比

| 指标  | Jittor (综合最佳) | PyTorch (综合最佳) | 差异（vs Pytorch） | Jittor优劣 |
| ----- | ----------------- | ------------------ | ------------------ | ---------- |
| IoU   | 0.5597            | 0.6201             | -9.74%             | ▼ 更差     |
| PD    | 0.0178            | 0.0177             | +0.56%             | ▲ 更优     |
| FA    | 18.8268           | 23.6094            | -20.26%            | ▲ 更优     |
| epoch | 46                | 51                 | -9.80%             | ▲ 更快     |

两个版本**metric.log**可视化及分析（图中标红处即为上表综合最佳指标）：

### jittor版：

<img src="https://github.com/Hareplace/MSHNet_jittor/blob/master/assert/jittor_metic_log.png"/>

### Jittor 版本指标曲线分析

#### **1. IoU over Epochs**

- 初始值为 0，快速上升至 0.48（epoch ≈ 10），此后增长缓慢。
- 最佳值出现在 **epoch = 46，IoU ≈ 0.5597**，但从 epoch ≈ 15 起增幅变缓，整体曲线较平滑，但未突破 0.6，存在早期饱和现象。

#### **2. PD over Epochs**

- 最初 PD 高达 0.6，下降非常迅速，到 **epoch ≈ 10 降至 0.1 以下**。
- **epoch ≈ 20 后稳定在 0.02 左右**，并在 epoch = 46 达到 0.0178。
- 下降趋势非常清晰，训练初期即可有效压低误检率。

#### **3. FA over Epochs (对数坐标)**

- **Jittor** 初始 FA 虽然远远低于 PyTorch，但下降过程更加“缓和”，**逐步下降到最低点 18.8（epoch 46）**，收敛过程稳定。
- 尽管部分中后期值略升高（如 epoch 66 为 45.1），但最终最佳点稳定在 **18.8**，且此前数个 epoch（如44）也维持在较低值（32.2）：**说明模型在后期能持续维持低误警率状态**。

### Pytorch版：

<img src="https://github.com/Hareplace/MSHNet_jittor/blob/master/assert/pytorch_metric_log.png"/>

###   PyTorch 版本指标曲线分析

#### **1. IoU over Epochs**

- 初期为 0，至 **epoch ≈ 15 快速上升并突破 0.5**，之后小幅振荡。
- 在 **epoch = 51 达到最高值 0.6201**，略优于 Jittor。
- 后期IoU提升幅度小，但整体稳定性好。

#### **2. PD over Epochs**

- 起始值同样为 0.6，至 **epoch ≈ 10 降至 0.1 以下**。
- 与 Jittor类似，最终收敛在 **0.0177（epoch = 51）**。
- 下降过程也非常迅速，几乎无差别。

#### **3. FA over Epochs (对数坐标)**

- **PyTorch** 的 FA 起始极高（数十万），但在 **短短几轮（6个epoch）内急剧下降到 20 左右**，说明模型极早过滤掉了大量误报，收敛速度非常快。
- 虽然初期下降快，但后期略有波动，FA 在 20~47 之间上下浮动（如 epoch 23 有 FA ≈ 43.6，epoch 34 ≈ 47.5），说明：**快速降低，但后期控制不如 Jittor 稳定**，最终选出的最佳 epoch（51）FA 为 **23.6094**，处于稳定但非最低。

#### 推论：

从指标曲线和日志数据分析可见，Jittor 和 PyTorch 两个版本的模型在训练过程中呈现出不同的收敛特性。**PyTorch 在 IoU 上达到更高峰值（0.6201），代表其在检测精度方面略优**；同时它在 FA 指标上也展现出极快的收敛能力——在极早期（第6个epoch）便将误警数从数十万迅速压至几十以下，体现出强劲的优化速度。然而，**PyTorch 的 FA 曲线在中后期存在一定波动，稳定性略逊**。

相较之下，**Jittor 的 FA 控制更加平稳**，虽然下降速度较缓，但最终在第46个epoch达到 **更低的 FA 最优值（18.8）**，并在多个 epoch 中维持在低误警水平。同时，其 loss 曲线波动小，训练稳定性更强，训练轮数也更少，说明其整体收敛过程更平滑。

综合考虑检测精度、误警控制方式、以及训练过程的稳定性，**Jittor 更适合部署在资源有限或对误报容忍度较低的实际系统中**，而 **PyTorch 更适合用于对指标极限有更高要求的研究场景或需要快速收敛的实验验证阶段**。

