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
    
#dropout_rates设置
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
results_dropout = {}

device = "cuda" if torch.cuda.is_available() else "cpu"


for rate in dropout_rates:
    model = DropoutFCNModel(input_size=3072, hidden_size=512, num_classes=10, dropout_rate=rate).to(device)
    basic_FCN_training_fc.train_model_FCN(model, CifarDataset.train_dataloader, num_epochs=10, learning_rate=0.001)
    accuracy = basic_FCN_test_fc.evaluate_accuracy(model, CifarDataset.test_dataloader)
    results_dropout[rate] = accuracy
    print(f'Dropout rate: {rate}, Test Accuracy: {accuracy:.2f}%')

# Optionally, to visualize the results:
import matplotlib.pyplot as plt
plt.plot(list(results_dropout.keys()), list(results_dropout.values()))
plt.xlabel('Dropout Rate')
plt.ylabel('Test Accuracy')
plt.title('Impact of Dropout Rate on Test Accuracy')
plt.show()