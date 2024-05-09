import torch
import torch.nn as nn



def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), images.size(2), -1).to(device)  #这里我一次输入整行像素（3*32）,reshape后images:(batch_size,32(时间步),32*3（送进去96个特征）)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


