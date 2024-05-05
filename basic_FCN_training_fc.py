import torch
import torch.optim as optim
import torch.nn as nn
import FCN_model

import CifarDataset

device="cuda" if torch.cuda.is_available() else "cpu"

def train_model_FCN(model, train_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad() # 清空梯度（如果不清空，梯度会累加）
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0: # 每100步打印一次
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

#FCN MODEL_TRAINING
model=FCN_model.FCNModel(3072,1000,10).to(device)
train_model_FCN(model, CifarDataset.train_dataloader, 5, 0.001)

# Save the model checkpoint
torch.save(model.state_dict(), 'machine_learning\HW04\FCN_model_basic_parameters\model_FCN.pth')