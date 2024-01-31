import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if label is None:
            return F.linear(input, self.weight)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

# 简单基础实现
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # Normalize input features
        x = F.normalize(input)
        # Normalize weights
        w = F.normalize(self.weight)

        # Cosine similarity between input and weights
        cosine = F.linear(x, w)

        # ArcFace margin
        phi = cosine - self.m
        one_hot = F.one_hot(label, num_classes=self.out_features).float()

        # Cosine with margin
        cosine_m = cosine * (1.0 - one_hot) + phi * one_hot

        # Scale the cosine values
        output = self.s * cosine_m

        return output

# 提供一个额外的全连接层fc
class ImprovedArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ImprovedArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.fc = nn.Linear(in_features, out_features)

    def forward(self, input, label=None):
        # Normalize input features
        x = F.normalize(input)
        # Normalize weights
        w = F.normalize(self.weight)

        # Cosine similarity between input and weights
        cosine = F.linear(x, w)

        # ArcFace margin
        phi = cosine - self.m
        one_hot = F.one_hot(label, num_classes=self.out_features).float()

        # Cosine with margin
        cosine_m = cosine * (1.0 - one_hot) + phi * one_hot

        # Scale the cosine values
        output = self.s * cosine_m

        # Additional fully connected layer for improved representation
        additional_output = self.fc(x)

        return output + additional_output

# 使用easy margin
class ImprovedArcFaceEasyMargin(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ImprovedArcFaceEasyMargin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.fc = nn.Linear(in_features, out_features)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # Normalize input features
        x = F.normalize(input)
        # Normalize weights
        w = F.normalize(self.weight)

        # Cosine similarity between input and weights
        cosine = F.linear(x, w)

        # ArcFace margin
        phi = cosine - self.m
        one_hot = F.one_hot(label, num_classes=self.out_features).float()

        # Cosine with margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Scale the cosine values
        output = self.s * (one_hot * phi + (1.0 - one_hot) * cosine)

        # Additional fully connected layer for improved representation
        additional_output = self.fc(x)

        return output + additional_output

# 使用ResNet残差块
class CustomNetwork(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(CustomNetwork, self).__init__()

        # CNN layers (ResNet-like block)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = self._make_res_block(64, 64, 2)  # Example: ResNet-like block with two residual layers

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, out_features)

        # ArcFace parameters
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = nn.Parameter(torch.cos(torch.tensor(m)))
        self.sin_m = nn.Parameter(torch.sin(torch.tensor(m)))
        self.th = nn.Parameter(torch.cos(torch.tensor(math.pi - m)))
        self.mm = nn.Parameter(torch.sin(torch.tensor(math.pi - m)) * m)

    def _make_res_block(self, in_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # Update for the next layer
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        # CNN forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_block(x)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # ArcFace margin
        cosine = F.linear(F.normalize(x), F.normalize(self.fc2.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # ArcFace output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

# 使用ResNet残差块的增强版---预训练resnet50
class AdvancedCustomNetwork(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(AdvancedCustomNetwork, self).__init__()

        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        # Modify the last fully connected layer to match the desired output features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_features)

        # ArcFace parameters
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = nn.Parameter(torch.cos(torch.tensor(m)))
        self.sin_m = nn.Parameter(torch.sin(torch.tensor(m)))
        self.th = nn.Parameter(torch.cos(torch.tensor(math.pi - m)))
        self.mm = nn.Parameter(torch.sin(torch.tensor(math.pi - m)) * m)

    def forward(self, x, label=None):
        # ResNet feature extraction
        features = self.resnet(x)

        # ArcFace margin
        cosine = F.linear(F.normalize(features), F.normalize(self.resnet.fc.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # ArcFace output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output