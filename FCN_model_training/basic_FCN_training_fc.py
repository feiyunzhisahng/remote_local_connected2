import torch
import torch.optim as optim
import torch.nn as nn
import FCN_model
import CifarDataset

device="cuda" if torch.cuda.is_available() else "cpu"

def train_model_FCN(model, train_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#gpt说：不需要将优化器移动到 GPU 上。优化器会自动操作已经在 GPU 上的模型参数
    model.train()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)#images已经在gpu上了，所以不需要再移动outputs到gpu上
            loss = criterion(outputs, labels)#labels,outputs都已经在gpu上了，所以不需要再移动loss到gpu上
            
            # Backward and optimize
            optimizer.zero_grad() # 清空梯度（如果不清空，梯度会累加）
            loss.backward()
            optimizer.step()
            
            # if (i+1) % 100 == 0: # 每100步打印一次
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

