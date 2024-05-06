import torch
import torch.optim as optim
import torch.nn as nn
import FCN_model
import basic_FCN_training_fc
import CifarDataset

device="cuda" if torch.cuda.is_available() else "cpu"

#FCN MODEL_TRAINING
model=FCN_model.FCNModel(3072,1000,10).to(device)
basic_FCN_training_fc.train_model_FCN(model, CifarDataset.train_dataloader, 5, 0.001)

# Save the model checkpoint
torch.save(model.state_dict(), 'remote_local_connected2\FCN_model_basic_parameters')