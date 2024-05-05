import torch
import torch.nn as nn
import torch.optim as optim
import FCN_model
import CifarDataset
import basic_FCN_training_fc
import basic_FCN_test

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

layer_configs = [(512,), (512, 256), (512, 256, 128), (512, 256, 128, 64)]
results_layers = {}

for config in layer_configs:
    model = ModifiedFCNModel(input_size=3072, hidden_sizes=config, num_classes=10).to(device)
    basic_FCN_training_fc.train_model_FCN(model, CifarDataset.train_dataloader, num_epochs=10, learning_rate=0.001)
    accuracy = basic_FCN_test.evaluate_accuracy(model, CifarDataset.test_dataloader)
    results_layers[config] = accuracy
    print(f'Layers config: {config}, Test Accuracy: {accuracy:.2f}%')
