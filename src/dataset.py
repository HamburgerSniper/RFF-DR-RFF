import os

import torch
import torch.utils.data.dataset

from .preprocessing import main as main_NMP
from .preprocessing_MP import main as main_MP
from .utils import TorchComplex as tc

"""
    FIR 滤波器实现：FIR滤波器是一种线性时不变的滤波器，其冲激响应在时域上是有限的,通过将输入信号与滤波器的系数进行线性卷积运算，对输入信号进行滤波处理。
    FIR滤波器常用于信号处理领域，可以根据需要对信号进行频率选择性滤波，去除其中的噪声和不需要的频率分量。它可以通过调整滤波器的系数来实现对信号的不同频率分量的强弱程度的控制。
    在射频指纹信号处理中，使用FIR滤波器主要考虑到其以下几种优点：
        (1)线性相位：在处理射频指纹信号时，我们需要保持信号的相位信息，以便于后续的分析和识别。FIR滤波器可以通过设计其系数，使其具有严格的线性相位特性，从而在滤波过程中不会引入额外的相位失真。
        (2)稳定特性：由于FIR滤波器的冲激响应是有限长的，因此它是一个稳定的系统，可以保证滤波结果的稳定输出。这对于射频指纹信号的处理尤为重要，因为我们需要稳定的信号来进行准确的分析和识别。
        (3)设计灵活性：FIR滤波器可以通过调整其系数来改变其频率响应特性，从而实现不同的滤波效果。这对于射频指纹信号的处理来说，可以根据实际需求灵活地设计滤波器，以达到最佳的滤波效果。
"""

"""
    x: 输入信号，其形状为 (batch_size, N, 2)，其中 N 是信号的长度，2 是信号的通道数（例如，对于QPSK调制，一个通道表示实部，另一个通道表示虚部）
    taps:  表示滤波器的阶数，默认值为 9。这意味着滤波器的冲击响应长度为 9
    padding: 添加到输入信号两端的零的数量，以确保滤波器可以覆盖整个信号
    x_pad: 添加了零填充的输入信号
    h: 滤波器的冲击响应，这里使用随机高斯分布生成，然后除以 (taps * 2) ** 0.5 以保证滤波器的功率为 1。
    x_list: 是一个列表，用于存储滤波器冲击响应与输入信号的卷积结果。
    X: 卷积结果的堆叠，形状为 (N, taps, 2)。
    x_FIR: 滤波器的输出，通过将 X 与 h 相乘得到。
    
    最后，x_FIR 被重新塑形为 (1, N, 2)，以便于返回
"""


# 定义FIR 接收参数为 输入信号x 和 滤波器阶数，默认为 9
def FIR(x, taps=9):
    # 获取输入信号x的形状，这里的形状为(batch_size, N, 2)，其中N是信号的长度，2是信号的通道数（例如，对于QPSK调制，一个通道表示实部，另一个通道表示虚部）
    _, N, _ = x.shape
    # 计算需要添加到输入信号两端的零的数量，以确保滤波器可以覆盖整个信号
    padding = int((taps - 1) / 2)
    # 添加零填充到输入信号的两端，以创建一个更大的信号，使得滤波器的冲击响应可以覆盖整个信号
    x_pad = torch.cat(
        [
            torch.zeros(padding, 2, dtype=x.dtype),
            x.view(N, 2),
            torch.zeros(padding, 2, dtype=x.dtype)
        ], dim=0)
    # 生成滤波器的冲击响应，这里使用随机高斯分布生成，然后除以 (taps * 2) ** 0.5 以保证滤波器的功率为 1
    h = torch.randn(taps, 1, 2, dtype=x.dtype) / (taps * 2) ** 0.5
    # 创建一个列表，用于存储滤波器冲击响应与输入信号的卷积结果
    x_list = []
    for i in range(N):
        x_list.append(x_pad[i:i + taps, :])
    X = torch.stack(x_list)
    # 将列表中的元素堆叠成一个张量，形状为(N, taps, 2)
    x_FIR = tc.mm(X, h)
    # 将滤波器的输出重塑为(1, N, 2)的形状，并返回
    return x_FIR.view(1, N, 2)


class RFdataset(torch.utils.data.Dataset):
    # 读取数据 -- device_ids: 一个列表 包含设备ids ; test_ids: 一个列表 包含测试ids ; flag: 一个字符串 表示数据集的来源 默认为'ZigBee'
    def __init__(self, device_ids, test_ids, flag='ZigBee', SNR=None, rand_max_SNR=None, is_FIR=False):
        if len(device_ids) > 1:
            device_flag = '{}-{}'.format(device_ids[0], device_ids[-1])
        else:
            device_flag = str(device_ids[0])

        test_flag = '-'.join([str(i) for i in test_ids])
        file_name = '{}_dv{}_id{}.pth'.format(flag, device_flag, test_flag)
        file_name = './datasets/processed/{}'.format(file_name)

        if not os.path.isfile(file_name):
            main_NMP(device_ids, test_ids, flag=flag)

        self.data = torch.load(file_name) # 数据集
        self.snr = SNR # 信噪比
        self.max_snr = rand_max_SNR # 随机生成的最大信噪比
        self.is_FIR = is_FIR # 是否需要对原始数据进行FIR滤波处理


    def __getitem__(self, index, x=None):
        idx = self.data['idx'][index]
        x_origin = self.data['x_origin'][index][idx:idx + 1280, :].view(1, -1, 2).clone().detach()
        x_syn = self.data['x_fopo'][index][idx:idx + 1280, :].view(1, -1, 2).clone().detach()
        y = self.data['y'][index]
        length = self.data['length'][index]

        # is_FIR为true则对原始数据进行FIR滤波处理
        if self.is_FIR:
            x_origin = FIR(x_origin)

        # 如果self.snr不为None，则对x_origin、x_syn和x添加高斯白噪声，模拟不同的信噪比（SNR）条件
        if not self.snr is None:
            x_origin += tc.awgn(x_origin, self.snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, self.snr, SNR_x=30)
            x += tc.awgn(x, self.snr, SNR_x=30)

        # 如果self.max_snr不为None，则随机生成一个介于5到se`lf.max_snr之间的SNR值，并添加到x_origin和x_syn中
        if not self.max_snr is None:
            rand_snr = torch.randint(5, self.max_snr, (1,)).item()
            x_origin += tc.awgn(x_origin, rand_snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, rand_snr, SNR_x=30)

        return x_origin, y, x_syn

    def __len__(self):
        return len(self.data['y'])


class RFdataset_MP(torch.utils.data.Dataset):
    def __init__(self, device_ids, test_ids, flag='ZigBee', SNR=None, rand_max_SNR=None):
        if len(device_ids) > 1:
            device_flag = '{}-{}'.format(device_ids[0], device_ids[-1])
        else:
            device_flag = str(device_ids[0])
        test_flag = '-'.join([str(i) for i in test_ids])
        file_name = '{}_dv{}_channel{}.pth'.format(flag, device_flag, test_flag)
        file_name = './datasets/processed/{}'.format(file_name)
        if not os.path.isfile(file_name):
            main_MP(device_ids, test_ids, flag=flag)
        self.data = torch.load(file_name)

        self.snr = SNR
        self.max_snr = rand_max_SNR

    def __getitem__(self, index):
        idx = self.data['idx'][index]
        x_origin = self.data['x_origin'][index][idx:idx + 1280, :].view(1, -1, 2).clone().detach()
        x_syn = self.data['x_fopo'][index][idx:idx + 1280, :].view(1, -1, 2).clone().detach()
        y = self.data['y'][index]
        length = self.data['length'][index]

        if not self.snr is None:
            x_origin += tc.awgn(x_origin, self.snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, self.snr, SNR_x=30)

        if not self.max_snr is None:
            rand_snr = torch.randint(5, self.max_snr, (1,)).item()
            x_origin += tc.awgn(x_origin, rand_snr, SNR_x=30)
            x_syn += tc.awgn(x_syn, rand_snr, SNR_x=30)
        return x_origin, y, x_syn

    def __len__(self):
        return len(self.data['y'])


if __name__ == "__main__":
    test = RFdataset_MP(device_ids=range(5), test_ids=[1, 2, 3], rand_max_SNR=None)
    print(len(test))
    print(test[0][0].shape)
    # min_freq = 1000000
    # max_freq = -1000000
    # sum_freq = 0.0
    # for i in range(len(test)):
    #     freq = test[i][0]
    #     if freq.max() > max_freq:
    #         max_freq = freq.max()
    #     if freq.min() < min_freq:
    #         min_freq = freq.min()
    #     sum_freq += freq.mean()
    # print(max_freq)
    # print(min_freq)
    # print(sum_freq/len(test))
