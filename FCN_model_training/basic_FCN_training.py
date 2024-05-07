import torch
import torch.optim as optim
import torch.nn as nn
import FCN_model
import basic_FCN_training_fc
import CifarDataset
import time
import multiprocessing

if __name__ == '__main__':#必须加这个判断，否则windows不能开启多进程
    multiprocessing.freeze_support()
    # 主程序代码
    device="cuda" if torch.cuda.is_available() else "cpu"

    start_time = time.time()


    #FCN MODEL_TRAINING
    model=FCN_model.FCNModel(3072,1000,10).to(device)
    basic_FCN_training_fc.train_model_FCN(model, CifarDataset.train_dataloader, 10, 0.001)
    excution_time = time.time() - start_time

    print("Training Completed")
    print(f"Execution time: {excution_time:.2f} seconds")

    # Save the model checkpoint
    torch.save(model.state_dict(), 'remote_local_connected2\FCN_model_basic_parameters')
