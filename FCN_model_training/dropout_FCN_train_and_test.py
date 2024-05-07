import torch
import basic_FCN_training_fc
import basic_FCN_test_fc
import CifarDataset
from dropout_FCN import DropoutFCNModel
import time


if __name__ == '__main__':
    #dropout_rates设置
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    results_dropout = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"


    for rate in dropout_rates:
        model = DropoutFCNModel(input_size=3072, hidden_sizes=[512,256,128], num_classes=10, dropout_rate=rate).to(device)
        start_time = time.time()
        basic_FCN_training_fc.train_model_FCN(model, CifarDataset.train_dataloader, num_epochs=10, learning_rate=0.001)
        traning_time = time.time() - start_time
        accuracy = basic_FCN_test_fc.evaluate_accuracy(model, CifarDataset.test_dataloader)
        results_dropout[rate] = accuracy
        print(f'Dropout rate: {rate}, Test Accuracy: {accuracy:.2f}%; Training time: {traning_time:.2f} seconds')
        

    # Optionally, to visualize the results:
    # import matplotlib.pyplot as plt
    # plt.plot(list(results_dropout.keys()), list(results_dropout.values()))
    # plt.xlabel('Dropout Rate')
    # plt.ylabel('Test Accuracy')
    # plt.title('Impact of Dropout Rate on Test Accuracy')
    # plt.show()