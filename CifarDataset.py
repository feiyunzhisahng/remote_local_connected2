import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms


class CifarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        Args:
            root_dir (str): 数据集的根目录。
            transform (callable, optional): 需要应用于样本的可选变换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        #self.files 每个元素是元组，元组的第一个元素是图像的路径，第二个元素是图像的标签
        self.files = []
        for cls in self.classes:
            cls_folder = os.path.join(self.root_dir, cls)
            for img in sorted(os.listdir(cls_folder)):
                self.files.append((os.path.join(cls_folder, img), cls))

    def __len__(self):
        """
        返回数据集中的样本数
        """
        return len(self.files)

    def __getitem__(self, index):
        """
        根据给定的索引返回样本
        Args:
            index (int): 样本的索引
        Returns:
            tuple: (image, label) 图像和其标签
        """
        img_path, cls = self.files[index]
        image = Image.open(img_path).convert('RGB')
        label = self.classes.index(cls)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# 数据增强和转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.0279, 0.1168, 0.1801], std=[0.8834, 0.8902, 0.8932])  # 归一化,数据来自calculate_mean_std.py
])

# 创建数据集
train_dataset = CifarDataset(root_dir='machine_learning\\HW04\\cifar10_train', transform=transform)

# 创建DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# 测试数据集
test_dataset = CifarDataset(root_dir='machine_learning\\HW04\\cifar10_test', transform=transform)

# 创建DataLoader
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)