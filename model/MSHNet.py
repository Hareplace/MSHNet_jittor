import jittor as jt
from jittor import nn
import jittor.nn as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc1 = nn.Conv(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):  # 注意 PyTorch 是 forward，Jittor 是 execute
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avg_out = jt.mean(x, dim=1, keepdims=True)
        max_out = jt.max(x, dim=1, keepdims=True)
        x = jt.concat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # 不要设为 None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
    def execute(self, x):
        residual = x
        if not isinstance(self.shortcut, nn.Identity):
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + residual
        out = self.relu(out)
        return out


class MSHNet(nn.Module):
    def __init__(self, input_channels, block=ResNet):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        self.pool = nn.Pool(kernel_size=2, stride=2, op="maximum")

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear')  

        self.conv_init = nn.Conv(input_channels, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block,
                                          param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block,
                                          param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block,
                                          param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block)

        self.output_0 = nn.Conv(param_channels[0], 1, 1)
        self.output_1 = nn.Conv(param_channels[1], 1, 1)
        self.output_2 = nn.Conv(param_channels[2], 1, 1)
        self.output_3 = nn.Conv(param_channels[3], 1, 1)

        self.final = nn.Conv(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def execute(self, x, warm_flag):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(jt.concat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(jt.concat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(jt.concat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(jt.concat([x_e0, self.up(x_d1)], 1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(jt.concat([
                mask0,
                self.up(mask1),
                self.up_4(mask2),
                self.up_8(mask3)
            ], dim=1))
            return [mask0, mask1, mask2, mask3], output

        else:
            output = self.output_0(x_d0)
            return [], output
