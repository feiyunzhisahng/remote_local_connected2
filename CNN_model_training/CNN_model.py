import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*16*16, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out
