import torch.nn as nn
import torch.optim as optim
import torch
import CifarDataset
import basic_FCN_training_fc
import basic_FCN_test_fc

class DropoutFCNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate):
        super(DropoutFCNModel, self).__init__()
        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_size, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # 添加 Dropout 层
            input_size = hs  # 更新输入大小为下一层的输出
        layers.append(nn.Linear(input_size, num_classes))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        return out
    
