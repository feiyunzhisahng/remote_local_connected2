import torch
from RNN_model import RNNModel
from RNN_test_fc import test_model
from CifarDataset import test_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"
#试一下16的准确度
Layer_dim=[16]

def main():
    for layer_dim in Layer_dim:
        model = RNNModel(input_dim=32*3, hidden_dim=128, layer_dim=layer_dim, output_dim=10, rnn_type='LSTM').to(device)
        model.load_state_dict(torch.load(f'remote_local_connected2\RNN_model_training\RNN_model_basic_parameters_LSTM/model_RNN_layer_dim_{layer_dim}.pth'))
        accuracy = test_model(model, test_dataloader, device=device)
        print(f'LSTC_Test Accuracy with layer_dim={layer_dim}: {accuracy:.2f}%')

        
        
        
if __name__=='__main__':
    main()