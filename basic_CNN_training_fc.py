import torch
import CNN_model
import CifarDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model_CNN(model, train_loader, num_epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    model.to(device)  # 确保模型在正确的设备上

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f'Average Loss for Epoch {epoch+1}: {total_loss / len(train_loader):.4f}')

# 实例化模型并训练
model = CNN_model.CNNModel(10).to(device)
train_model_CNN(model, CifarDataset.train_dataloader, 5, 0.001)

# 保存训练后的模型
torch.save(model.state_dict(), 'machine_learning\HW04\CNN_model_basic_parameters\model_CNN.pth')
