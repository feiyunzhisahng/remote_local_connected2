import torch
import CNN_model
import basic_CNN_test
import CifarDataset
import basic_CNN_test

batch_sizes = [16, 32, 64, 128]  # 不同的批次大小
results = {}

device="cuda" if torch.cuda.is_available() else "cpu"

for batch_size in batch_sizes:
    train_loader = torch.utils.data.DataLoader(CifarDataset.train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(CifarDataset.test_dataset, batch_size=batch_size, shuffle=False)
    model = CNN_model.CNNmodel(num_classes=10).to(device)
    basic_CNN_test.train_model(model, train_loader, num_epochs=10, learning_rate=0.001)
    accuracy = basic_CNN_test.evaluate_accuracy(model, test_loader)
    results[batch_size] = accuracy
    print(f'Batch size: {batch_size}, Test Accuracy: {accuracy:.2f}%')

print(results)
