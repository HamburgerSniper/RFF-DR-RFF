import sys
sys.path.append("/home/liuxuanchen/codings/pythonproject/RFF-DR-RFF/src/")

is_debug = False
if not is_debug:
    from ZigBee_processing import *
    from ArcFace import *
else:
    from ZigBee_processing import *
    from ArcFace import *

# 归一化模型
class NormalizedModel(nn.Module):
    def __init__(self) -> None:
        super(NormalizedModel, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # mean = input.mean(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        # std = input.std(dim=2, keepdim=True).repeat(1, 1, 1280, 1)
        # 计算均值mean和标准差std
        mean = input.mean()
        std = input.std()
        normalized_input = (input - mean) / std
        return normalized_input

# 分类器
# 归一化 + 卷积/池化/激活 * 3 + 全连接层 输出output为10个类别
class BaseCLF(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, d=64):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (1x1280x2)
            NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=2 * d, kernel_size=(10, 1), stride=1, padding=(5, 0)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4, 1)),

            # State (128x320x2)
            nn.Conv2d(in_channels=2 * d, out_channels=4 * d, kernel_size=(3, 2), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4, 1)),

            # State (256x80x3)
            nn.Conv2d(in_channels=4 * d, out_channels=4 * d, kernel_size=(80, 3), stride=1, padding=0),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
        )
        # outptut of main module --> State (1024x4x4)

        self.output = nn.Linear(4 * d, out_channels)

    def forward(self, input):
        out = self.features(input)
        out = self.output(out)
        return out

    def features(self, input):
        N = len(input)
        out = self.main_module(input).view(N, -1)
        return out

# 分类器
# 取消全连接层 通过卷积层直接产生输出分类，更适用于时间序列数据
class BaseCLF2(nn.Module):
    def __init__(self, in_channels=2, out_dim=1, d=4):
        super().__init__()
        self.d = d
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d * 2, out_channels=self.d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d * 8, out_channels=self.d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d * 32, out_channels=out_dim, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            # nn.BatchNorm2d(out_dim),

        )
        # outptut of main module --> State (1024x4x4)
        # self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        out = self.features(input)
        # out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        input_img = input.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(input_img).view(N, -1)
        return out

# resblock + attention
class BaseCLF4(nn.Module):
    def __init__(self, in_channels=2, out_dim=1, d=4):
        super().__init__()
        self.d = d

        # Define residual block
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 4, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 4, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4)
        )

        # Define attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 16, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.Softmax(dim=2)
        )

        self.main_module = nn.Sequential(
            # Image (2x16x80)
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d * 2, out_channels=self.d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d * 8, out_channels=self.d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d * 32, out_channels=self.d * 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # Residual block
            self.resblock,

            # Attention mechanism
            self.attention,

            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(self.d * 16, out_dim)

    def forward(self, input, labels=None):
        out = self.features(input)
        out = self.fc(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        input_img = input.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(input_img).view(N, -1)
        return out

# 分类器
# 全连接层分类
class BaseCLF3(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, d=4):
        super().__init__()
        self.d = d
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d * 2, out_channels=self.d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d * 8, out_channels=self.d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            # nn.BatchNorm2d(self.d*32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d * 32, out_channels=512, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            # nn.BatchNorm2d(512),
            nn.Dropout(p=0.5)
        )
        # outptut of main module --> State (1024x4x4)
        self.output = nn.Linear(512, out_channels)

    def forward(self, input, labels=None):
        out = self.features(input)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        input_img = input.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(input_img).view(N, -1)
        return out

# residual block + attention mechanism + transformer + global average pooling
class BaseCLF5(nn.Module):
    def __init__(self, in_channels=2, out_dim=1, d=4):
        super().__init__()
        self.d = d

        # Define residual block
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 4, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 4, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4)
        )

        # Define attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 16, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.Softmax(dim=2)
        )

        self.main_module = nn.Sequential(
            # Image (2x16x80)
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d * 2, out_channels=self.d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d * 8, out_channels=self.d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d * 32, out_channels=self.d * 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # Residual block
            self.resblock,

            # Attention mechanism
            self.attention,

            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(self.d * 16, out_dim)

    def forward(self, input, labels=None):
        out = self.features(input)
        out = self.fc(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        input_img = input.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(input_img).view(N, -1)
        return out


# 频率预处理 - 频偏估计/频率补偿
class Freq_processing(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_offset_estimation = BaseCLF5(1, out_channels=1)

    def forward(self, input):
        N, _, T, _ = input.shape
        freq = self.freq_offset_estimation(input)
        out = freq_compensation(input.view(N, T, -1), freq.view(-1))
        return out, freq

# 相位预处理 - 相位偏移估计/相位补偿
class Phase_processing(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase_offset_estimation = BaseCLF5(1, out_channels=1)

    def forward(self, input):
        N, _, T, _ = input.shape
        phase = self.phase_offset_estimation(input)
        out = phase_compensation(input.view(N, T, -1), phase.view(-1))
        return out, phase

class Synchronization(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.freq_estimation = BaseCLF2(2, out_dim=1, d=d)
        self.phase_estimation = BaseCLF2(2, out_dim=1, d=d)

    def forward(self, input):
        N, _, T, _ = input.shape
        freq_offset = self.freq_estimation(input.view(N, 1, T, 2)).view(-1)
        out = freq_compensation(input.view(N, T, -1), freq_offset)
        phase_offset = self.phase_estimation(out.view(N, 1, T, 2)).view(-1)
        out = phase_compensation(out.view(N, T, -1), phase_offset)
        return out.view(N, 1, T, 2), freq_offset, phase_offset

class SynchronizationVis(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.freq_estimation = BaseCLF(1, out_channels=1, d=d)
        self.phase_estimation = BaseCLF(1, out_channels=1, d=d)

    def forward(self, input):
        N, _, T, _ = input.shape
        freq_offset = self.freq_estimation(input.view(N, 1, T, 2)).view(-1)
        out = freq_compensation(input.view(N, T, -1), freq_offset)
        phase_offset = self.phase_estimation(out.view(N, 1, T, 2)).view(-1)
        out = phase_compensation(out.view(N, T, -1), phase_offset)
        return out.view(N, 1, T, 2), freq_offset, phase_offset

###########################
class CLF_yjb(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (1x1280x2)
            NormalizedModel(),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(10, 1), stride=1, padding=(5, 0)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4, 1)),

            # State (128x320x2)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 2), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((4, 1)),

            # State (256x80x3)
            nn.Conv2d(in_channels=256, out_channels=z_dim, kernel_size=(80, 3), stride=1, padding=0),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
        )
        # outptut of main module --> State (1024x4x4)

        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        out = self.features(input)
        out = self.output(out)
        return out

    def features(self, input):
        N = len(input)
        out = self.main_module(input).view(N, -1)
        return out

class NS_CLF_Arcface_old(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.freq_processing = Freq_processing()
        self.phase_processing = Phase_processing()

        self.d = 32
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d * 2, out_channels=self.d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d * 8, out_channels=self.d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d * 32, out_channels=z_dim, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(z_dim),

        )
        # outptut of main module --> State (1024x4x4)

        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.5)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment1, freq1 = self.freq_processing(input.view(N, 1, T, 2))
        segment2, phase = self.phase_processing(segment1.view(N, 1, T, 2))
        nn_input = segment2.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(nn_input).view(N, -1)
        return out

class NS_CLF_Softmax_old(nn.Module):
    def __init__(self, in_channels=2, out_channels=10, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.freq_processing = Freq_processing()
        self.phase_processing = Phase_processing()

        self.d = 32
        self.main_module = nn.Sequential(
            # Image (2x16x80)
            # NormalizedModel(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.d, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2),

            # State (dx16x80)
            nn.Conv2d(in_channels=self.d, out_channels=self.d * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 2),
            nn.LeakyReLU(0.2),

            # State (2dx16x80)
            nn.Conv2d(in_channels=self.d * 2, out_channels=self.d * 4, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 4),
            nn.LeakyReLU(0.2),

            # State (4dx8x40)
            nn.Conv2d(in_channels=self.d * 4, out_channels=self.d * 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 8),
            nn.LeakyReLU(0.2),

            # State (8dx8x40)
            nn.Conv2d(in_channels=self.d * 8, out_channels=self.d * 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 16),
            nn.LeakyReLU(0.2),

            # State (16dx4x20)
            nn.Conv2d(in_channels=self.d * 16, out_channels=self.d * 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(self.d * 32),
            nn.LeakyReLU(0.2),

            # State (32dx2x10)
            nn.Conv2d(in_channels=self.d * 32, out_channels=z_dim, kernel_size=(2, 10), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(z_dim),

        )
        # outptut of main module --> State (1024x4x4)

        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment1, freq1 = self.freq_processing(input.view(N, 1, T, 2))
        segment2, phase = self.phase_processing(segment1.view(N, 1, T, 2))
        nn_input = segment2.view(N, 1, T, 2).permute(0, 3, 1, 2).flatten().view(N, -1, 16, 80)
        out = self.main_module(nn_input).view(N, -1)
        return out

# 非HP 且 非NS
class CLF_Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=32, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment = input
        out = self.main_module.features(segment).view(N, -1)
        return out

# HP 且 非NS
class CLF_L2Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.0)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment = input
        out = self.main_module.features(segment).view(N, -1)
        return out


class CLF_Arcface(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.5)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment = input
        out = self.main_module.features(segment).view(N, -1)
        return out

# 非HP 且 NS
class NS_CLF_Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = Synchronization(d=d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_Softmax_Vis(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = SynchronizationVis(d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = nn.Linear(z_dim, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

# HP 且 NS
class NS_CLF_L2Softmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = Synchronization(d=d1)
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.0)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_L2Softmax_Vis(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = SynchronizationVis()
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.0)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

class NS_CLF_Arcface(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512):
        super().__init__()
        # Input_dim = channels (1x1280x2)
        # Output_dim = 1
        self.synchronization = Synchronization(d=d1)

        self.d = 16
        self.main_module = BaseCLF2(2, out_dim=z_dim, d=d2)
        self.output = ArcMarginProduct(z_dim, out_channels, s=10, m=0.5)
        # self.output = nn.Linear(512, out_channels)

    def forward(self, input, labels=None):
        segment = input
        out = self.features(segment)
        out = self.output(out, labels)
        return out

    def features(self, input):
        N, _, T, _ = input.shape
        segment, _, _ = self.synchronization(input)
        out = self.main_module.features(segment).view(N, -1)
        return out

# 生成器
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1280*2):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Tanh激活函数使输出范围在-1到1之间
        )

    def forward(self, z):
        # z是输入的随机噪声向量
        output = self.model(z)
        # 将输出调整为与RFdataset数据格式相同的形状
        output = output.view(-1, 1, 1280, 2)
        return output

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=1280*2):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 将输入展平
        x = x.view(x.size(0), -1)
        output = self.model(x)
        return output


if __name__ == '__main__':
    x = torch.randn(10, 1, 1280, 2)
    test_model = NS_CLF_Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)
    test_model = NS_CLF_L2Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)
    test_model = CLF_L2Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)
    test_model = CLF_Softmax(in_channels=3, out_channels=10, d1=8, d2=24, z_dim=512)
    print(test_model(x).shape)
