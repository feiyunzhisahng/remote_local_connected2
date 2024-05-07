import torch
import FCN_model
import CifarDataset
import basic_FCN_test_fc
import time



start_time = time.time()



device="cuda" if torch.cuda.is_available() else "cpu"



model=FCN_model.FCNModel(3072,1000,10).to(device)
model.load_state_dict(torch.load('remote_local_connected2\FCN_model_training\FCN_model_basic_parameters\model_FCN.pth'))
test_accuracy = basic_FCN_test_fc.evaluate_accuracy(model, CifarDataset.test_dataloader)



print(f'Test Accuracy: {test_accuracy:.2f}%')
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")