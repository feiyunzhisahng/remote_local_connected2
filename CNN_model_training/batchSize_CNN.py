import torch
import CNN_model
import basic_CNN_test_fc
import CifarDataset
import basic_CNN_training_fc
import time

batch_sizes = [16, 32, 64, 128]  # 不同的批次大小
results = {}

device="cuda" if torch.cuda.is_available() else "cpu"

for batch_size in batch_sizes:
    
    start_time = time.time()
    
    train_loader = torch.utils.data.DataLoader(CifarDataset.train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(CifarDataset.test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CNN_model.CNNModel(num_classes=10).to(device)
    basic_CNN_training_fc.train_model_CNN(model, train_loader, num_epochs=10, learning_rate=0.001)
    
    accuracy = basic_CNN_test_fc.evaluate_accuracy(model, test_loader)
    
    results[batch_size] = accuracy
    print(f'Batch size: {batch_size}, Test Accuracy: {accuracy:.2f}%')
    
    excution_time = time.time() - start_time
    print(f" Execution time: {excution_time:.2f} seconds")

print(results)
