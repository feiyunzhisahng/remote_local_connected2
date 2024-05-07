import torch
import torch.nn as nn
import CifarDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 32, 32).to(device)  # Reshape images to (batch_size, seq_length, input_size)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Assuming the use of CifarDataset for loading data
# Assuming train_loader and test_loader are set up properly
