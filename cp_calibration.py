import yaml
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataloader import  SolarFlSets
from training.trainloop import test_loop

from training.models import get_model
from training.loss import QuantileLoss
from posthoc_cp.calibration import ConformalPredictor


if __name__ == "__main__":

    # Load config file
    config_path = "./configs/config_cp.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    if isinstance(config["cuda"]["device"], int):
        device = torch.device(f"cuda:{config['cuda']['device']}" if use_cuda else "cpu")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print("1st check cuda..")
    print("Number of available device", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device:", device)

    # dataset partitions and create data frame
    print("2nd process, loading data...")

    # define directory here
    img_dir = config['dir']['img_dir']
    index_dir = config['dir']['index_dir']
    save_dir = config['dir']['save_dir']
    model_name = config['model']['name']
    mode = config['model']['mode']

    # print out experiment setting
    print(f"Model: {config['model']['name']}")
    for key, value in config['optimize'].items():
        print(f"{key}: {value}")

    model_files = glob.glob(save_dir + f"/model/{model_name}*{mode}*.pth")
    
    if len(model_files) == 1:
        # Load the state_dict
        checkpoint = torch.load(model_files[0], map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
    else:
        print("Check pth files!")

    # Remove 'module.' prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Strip 'module.' prefix
        else:
            new_state_dict[k] = v

    model = get_model(config['model']).to(device)
    # Load the modified state_dict
    model.load_state_dict(new_state_dict)

    df_cal_list = [pd.read_csv(index_dir + '/' + file) for file in config['dir']['cal_p']]
    df_cal = pd.concat(df_cal_list, ignore_index=True)
    df_cal["Timestamp"] = pd.to_datetime(df_cal["Timestamp"], format="%Y-%m-%d %H:%M:%S")

    data_cal = SolarFlSets(annotations_df=df_cal, img_dir=img_dir, normalization=True) #, target_transform=True
    cal_dataloader = DataLoader(data_cal, batch_size=config["optimize"]["batch_size"], shuffle=False)

    loss_fn = nn.MSELoss() if mode == 'cp' else QuantileLoss(quantile_val = config['model']['q_val'])
    cal_loss, cal_dict = test_loop(
                cal_dataloader, model=model, loss_fn=loss_fn
            )

    # load test results
    log_files = glob.glob(save_dir + f"/log/{model_name}*{mode}*.npz")
    if len(log_files) == 1:
        # Load prediction results using np.load()
        result = np.load(log_files[0], allow_pickle=True)
        val_result = result["test"].item()
    elif len(log_files) > 1:
        print("More than 1 files are identified.")
    else:
        raise("The file does not exist. Please check npz file.")
    

    # ------------------------ conformal prediction ------------------------------------------
    
    # Loop through confidence levels
    start = 0
    end = 1.0
    interval = 0.05
    
    cp_list = []
    mcp_list = []
    for conf in np.arange(start, end, interval):
        print(f"Processing confidence: {conf*100:.1f}%...")

        # Create a new CP instance
        CP = ConformalPredictor(
            error = 1 - conf, 
            mondrian = False
        )
        
        MCP = ConformalPredictor(
            error = 1 - conf, 
            bins = [0, 1, 2, 3, np.inf],
            mondrian = True
        )

        # Configure based on approach type
        CP.q_score(
            cal_arr = cal_dict['prediction'], 
            label = cal_dict['label'], 
            mode = mode
        )

        print(f"q_hat: {CP.q_hat}")
        
        cp_list.append(CP.pred_regions(test_arr = val_result['prediction']))

        MCP.q_score(
            cal_arr = cal_dict['prediction'], 
            label = cal_dict['label'], 
            mode = mode
        )

        print(f"q_hat: {MCP.q_hat_dict}")

        mcp_list.append(MCP.pred_regions(test_arr = val_result['prediction']))



    np.savez(
    f'./results/uncertainty/{model_name}_{mode}_AllRegions.npz',
    cp=np.stack(cp_list),
    mcp=np.stack(mcp_list),
    val=val_result['prediction'],
    label=val_result['label']
    )
