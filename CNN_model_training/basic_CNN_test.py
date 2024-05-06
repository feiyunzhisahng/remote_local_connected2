import torch
import CNN_model
import CifarDataset
import basic_CNN_test_fc
import time



device = "cuda" if torch.cuda.is_available() else "cpu"
start_time = time.time()


model=CNN_model.CNNModel(10).to(device)
model.load_state_dict(torch.load('remote_local_connected2\CNN_model_training\CNN_model_basic_parameters\model_CNN.pth'))
test_accuracy = basic_CNN_test_fc.evaluate_accuracy(model, CifarDataset.test_dataloader)


print(f'Test Accuracy: {test_accuracy:.2f}%')

excution_time = time.time() - start_time
print(f"Execution time: {excution_time:.2f} seconds")