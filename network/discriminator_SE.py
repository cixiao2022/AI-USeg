import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # 计算重要性权重
        attention = nn.functional.sigmoid(self.conv(x))

        # 将关注度应用于输入特性图
        return attention * x



class FCDiscriminator(nn.Module):
    def __init__(self, num_classes=19, ndf=64):
        super(FCDiscriminator, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)

        # 定义注意力层
        self.attention3 = Attention(ndf * 4)
        self.attention4 = Attention(ndf * 4)

        # 定义激活函数
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # 应用卷积操作和激活函数
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.attention3(x)  # 应用注意力层在 conv3 输出上
        x = self.activation(x)
        x = self.conv4(x)
        x = self.attention4(x)  # 应用注意力层在 conv4 输出上
        x = self.activation(x)
        x = self.classifier(x)

        return x