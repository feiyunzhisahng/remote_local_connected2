import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import CifarDataset

# 定义转换为Tensor的转换
transform = transforms.Compose([transforms.ToTensor()])


def calculate_mean_std(loader):
    # 通过迭代整个数据集来计算平均值和标准差
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # 批次中的样本数
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    return mean, std

"""
import torch
from torch.serialization import load
import torchvision.datasets as datasets
import torchvision.transforms as tansformes
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

train_dataset = datasets.CIFAR10(root="CIFAR/",train=True,transform=tansformes.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum,channels_squared_sum,num_batches = 0,0,0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    print(num_batches)
    print(channels_sum)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) **0.5

    return mean,std

mean,std = get_mean_std(train_loader)

print(mean)
print(std)

"""

mean, std = calculate_mean_std(CifarDataset.train_dataloader)

print(f'Mean: {mean}')
print(f'Std: {std}')
