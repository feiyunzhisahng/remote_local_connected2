import torch
import torch.nn as nn
import torch.optim as optim
import FCN_model
import CifarDataset
import basic_FCN_training_fc
import basic_FCN_test_fc

device = "cuda" if torch.cuda.is_available() else "cpu"

class ModifiedFCNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(ModifiedFCNModel, self).__init__()
        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_size, hs))
            layers.append(nn.ReLU())
            input_size = hs  # 更新输入大小为下一层的输出
        layers.append(nn.Linear(input_size, num_classes))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        return out


# 隐藏层配置
layer_configs_1 = [(512,), (512, 256), (512, 256, 128), (512, 256, 128, 64)]
layer_configs_2 = [(256,), (256, 128), (256, 128, 64),  (256, 128,  64,  32)]

results_layers = {}

for config1 in layer_configs_1:
    model = ModifiedFCNModel(input_size=3072, hidden_sizes=config1, num_classes=10).to(device)
    basic_FCN_training_fc.train_model_FCN(model, CifarDataset.train_dataloader, num_epochs=10, learning_rate=0.001)
    accuracy = basic_FCN_test_fc.evaluate_accuracy(model, CifarDataset.test_dataloader)
    results_layers[config1] = accuracy
    print(f'Layers config_1: {config1}, Test Accuracy: {accuracy:.2f}%')

for config2 in layer_configs_2:
    model = ModifiedFCNModel(input_size=3072, hidden_sizes=config2, num_classes=10).to(device)
    basic_FCN_training_fc.train_model_FCN(model, CifarDataset.train_dataloader, num_epochs=10, learning_rate=0.001)
    accuracy = basic_FCN_test_fc.evaluate_accuracy(model, CifarDataset.test_dataloader)
    results_layers[config2] = accuracy
    print(f'Layers config_2: {config2}, Test Accuracy: {accuracy:.2f}%')

print(results_layers)