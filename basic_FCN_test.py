import torch
import FCN_model
import CifarDataset

device="cuda" if torch.cuda.is_available() else "cpu"

def evaluate_accuracy(model, data_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


model=FCN_model.FCNModel(3072,1000,10).to(device)
model.load_state_dict(torch.load('machine_learning\HW04\FCN_model_basic_parameters\model_FCN.pth'))
test_accuracy = evaluate_accuracy(model, CifarDataset.test_dataloader)
print(f'Test Accuracy: {test_accuracy:.2f}%')
