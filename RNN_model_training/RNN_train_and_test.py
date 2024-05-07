import torch
from RNN_model import RNNModel
from RNN_training_fc import train_model
from RNN_test_fc import test_model
from CifarDataset import train_dataloader, test_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

# 运行实验
model = RNNModel(input_dim=32, hidden_dim=128, layer_dim=1, output_dim=10, rnn_type='LSTM').to(device)
train_model(model, train_dataloader, num_epochs=10, learning_rate=0.001,device=device)
accuracy = test_model(model, test_dataloader)
print(f'Test Accuracy of the model on the 10000 test images: {accuracy:.2f}%')