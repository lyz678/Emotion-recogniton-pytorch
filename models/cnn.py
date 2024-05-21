import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MiniXception(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MiniXception, self).__init__()
        # base
        self.conv1 = nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)

        # module 1
        self.residual1 = nn.Conv2d(8, 16, kernel_size=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv3 = SeparableConv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv4 = SeparableConv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(16)

        # module 2
        self.residual2 = nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv5 = SeparableConv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv6 = SeparableConv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(32)

        # module 3
        self.residual3 = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv7 = SeparableConv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv8 = SeparableConv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(64)

        # module 4
        self.residual4 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv9 = SeparableConv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv10 = SeparableConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # base
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # module 1
        residual1 = self.residual1(x)
        residual1 = self.bn3(residual1)

        x = F.relu(self.bn4(self.conv3(x)))
        x = F.relu(self.bn5(self.conv4(x)))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x += residual1

        # module 2
        residual2 = self.residual2(x)
        residual2 = self.bn6(residual2)

        x = F.relu(self.bn7(self.conv5(x)))
        x = F.relu(self.bn8(self.conv6(x)))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x += residual2

        # module 3
        residual3 = self.residual3(x)
        residual3 = self.bn9(residual3)

        x = F.relu(self.bn10(self.conv7(x)))
        x = F.relu(self.bn11(self.conv8(x)))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x += residual3

        # module 4
        residual4 = self.residual4(x)
        residual4 = self.bn12(residual4)

        x = F.relu(self.bn13(self.conv9(x)))
        x = F.relu(self.bn14(self.conv10(x)))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x += residual4

        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.softmax(x, dim=1)


class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
 
        '''
        如果步长 stride 不为 1 或者输入通道数 in_planes 不等于扩展系数 self.expansion 乘以输出通道数 planes，
        则需要进行维度匹配，使用一个包含一个卷积层和一个批量归一化层的序列 self.shortcut 来进行匹配
        '''
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, input_shape, num_classes, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.softmax(out, dim=1)
        return out
 
 
def ResNet18(input_shape, num_classes):
    return ResNet(input_shape, num_classes, BasicBlock, [2, 2, 2, 2])
 
 
def ResNet34(input_shape, num_classes):
    return ResNet(input_shape, num_classes, BasicBlock, [3, 4, 6, 3])
 
 
def ResNet50(input_shape, num_classes):
    return ResNet(input_shape, num_classes, Bottleneck, [3, 4, 6, 3])
 
 
def ResNet101(input_shape, num_classes):
    return ResNet(input_shape, num_classes, Bottleneck, [3, 4, 23, 3])
 
 
def ResNet152(input_shape, num_classes):
    return ResNet(input_shape, num_classes, Bottleneck, [3, 8, 36, 3])
 


if __name__ == "__main__":
    input_shape = (1, 48, 48)
    num_classes = 7
    model = MiniXception(input_shape, num_classes)
    print(model)
