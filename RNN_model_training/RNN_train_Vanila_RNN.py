import torch
from RNN_model import RNNModel
from RNN_training_fc import train_model
from RNN_test_fc import test_model
from CifarDataset import train_dataloader, test_dataloader
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 创建保存参数的目录
    output_dir = 'remote_local_connected2\RNN_model_training\RNN_model_basic_parameters_RNN'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    #layer_dim=[1,2,3]
    Layer_dim=[1,2,3]

    
    # 运行实验
    for layer_dim in Layer_dim:
        model = RNNModel(input_dim=32*3, hidden_dim=128, layer_dim=layer_dim, output_dim=10, rnn_type='RNN').to(device)
        start_time = time.time()
        train_model(model, train_dataloader, num_epochs=10, learning_rate=0.001,device=device)
        excution_time = time.time()-start_time
        model_path = os.path.join(output_dir, f'model_RNN_layer_dim_{layer_dim}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path};training time: {excution_time:.2f}s')
        

if __name__ == '__main__':
    main()
    

