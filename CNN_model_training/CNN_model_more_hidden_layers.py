import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, num_classes, hidden_layers=None):
        super(CNNModel, self).__init__()
        # 初始化层列表
        layers = []
        
        # 添加初始卷积层
        in_channels = 3  # 输入图像是RGB, 所以有3个通道
        # 定义第一个卷积层
        layers += [nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=2, stride=2)]
        
        # 动态添加更多卷积层
        current_channels = 16
        if hidden_layers:
            for hl in hidden_layers:
                layers += [nn.Conv2d(current_channels, hl['out_channels'],
                                     kernel_size=hl.get('kernel_size', 5),
                                     stride=hl.get('stride', 1),
                                     padding=hl.get('padding', 2)),#默认值
                           nn.ReLU()]
                if 'maxpool' in hl and hl['maxpool']:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                current_channels = hl['out_channels']

        # 将列表转换为nn.Sequential模型
        self.layers = nn.Sequential(*layers)
        
         # 使用一个模拟输入来计算卷积层输出的维度
        with torch.no_grad():
            sample_output = self.layers(torch.zeros(1, 3, 32, 32))  # 假设输入尺寸为32x32
            flattened_size = sample_output.view(1, -1).size(1)
        
        # 创建全连接层
        self.fc = nn.Linear(flattened_size, num_classes)
    
    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out


# 隐藏层配置的一种方式
cnn_config = [
    {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'maxpool': True},  # 第一层，带池化
    {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'maxpool': True},  # 第二层，带池化
    {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'maxpool': False}, # 第三层，不带池化
    {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'maxpool': True}  # 第四层，带池化
]



