import basic_CNN_training_fc
import torch
import CNN_model
import CifarDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# 实例化模型并训练
model = CNN_model.CNNModel(10).to(device)
basic_CNN_training_fc.train_model_CNN(model, CifarDataset.train_dataloader, 10, 0.001)

# 保存训练后的模型
torch.save(model.state_dict(), 'remote_local_connected2\basic_CNN_model_training\CNN_model_basic_parameters')
