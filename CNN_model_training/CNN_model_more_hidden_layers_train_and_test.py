import basic_CNN_training_fc
from CNN_model_more_hidden_layers import CNNModel, cnn_config
import basic_CNN_test_fc
import CifarDataset
import time

model = CNNModel(num_classes=10, hidden_layers=cnn_config)

start_time = time.time()

#train
basic_CNN_training_fc.train_model_CNN(model, CifarDataset.train_dataloader ,num_epochs=10, learning_rate=0.001)
excution_time = time.time() - start_time
print(f'Execution time: {excution_time:.2f} seconds;Training completed')

#test
accuracy = basic_CNN_test_fc.evaluate_accuracy(model, CifarDataset.test_dataloader)
print(f'Test Accuracy: {accuracy:.2f}%')





