# basic package
import os
# import glob
# import json
import yaml
import time
import datetime
# import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# pytorch package
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# predefined class
from training.loss import quantile_loss
from training.models import get_model
from training.dataloader import  SolarFlSets
from training.trainloop import train_loop, test_loop

if __name__ == "__main__":

    # Load config file
    config_path = "./configs/config_cp.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # CUDA for PyTorch
    
    use_cuda = torch.cuda.is_available()
    if isinstance(config["cuda"]["device"], int):  
        device = torch.device("cuda: {config.cuda.device}" if use_cuda else "cpu")
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

    # print out experiment setting
    print(f"Model: {config['model']['name']}")
    for key, value in config['optimize'].items():
        print(key, value)

    # Define dataset here!
    # train/test set
    df_train = pd.read_csv(
        os.path.join(index_dir, "24image_reg_train.csv")
    )
    df_test = pd.read_csv(
        os.path.join(index_dir, "24image_reg_test.csv")
    )

    # # string to datetime
    df_train["Timestamp"] = pd.to_datetime(df_train["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    df_test["Timestamp"] = pd.to_datetime(df_test["Timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Define dataset
    data_train = SolarFlSets(annotations_df=df_train, img_dir=img_dir, normalization=True)
    data_test = SolarFlSets(annotations_df=df_test, img_dir=img_dir, normalization=True)

    # Data loader
    train_dataloader = DataLoader(data_train, batch_size=config["optimize"]["batch_size"], shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=config["optimize"]["batch_size"], shuffle=False)

    
    """
        [ Grid search start here ]
        - Be careful with result array, model, loss, and optimizer
        - Their position matters
    """
    best_loss = float("inf")
    best_epoch = 0
    training_result = []
    for wt in config['optimize']['wt_decay']:
        
        net = get_model(config['model'])
        model = nn.DataParallel(net, device_ids=config['cuda']['device']).to(device) # multiple devices

        device = next(model.parameters()).device
        loss_fn = nn.MSELoss() if config['model']['mode'] == 'cp' else quantile_loss(quantile_val = config['model']['q_val'])
        optimizer = torch.optim.SGD(model.parameters(), lr=config['optimize']['lr'], weight_decay=wt)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            steps_per_epoch=len(train_dataloader),
            anneal_strategy="cos",
            pct_start=0.7,
            **config['scheduler']
        )

        # initiate variable for finding best epoch

        for t in range(config['scheduler']['epochs']):

            # extract current time and compute training time
            t0 = time.time()
            datetime_object = datetime.datetime.fromtimestamp(t0)
            year = datetime_object.year
            month = datetime_object.month

            train_loss, train_dict = train_loop(
                train_dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=scheduler
            )
            test_loss, test_dict = test_loop(
                test_dataloader, model=model, loss_fn=loss_fn
            )
            train_r2 = r2_score(train_dict['label'], train_dict['prediction']) # r2_score(y_true, y_pred)
            test_r2 = r2_score(test_dict['label'], test_dict['prediction']) # r2_score(y_true, y_pred)

            # trace score and predictions
            duration = (time.time() - t0) / 60
            actual_lr = optimizer.param_groups[0]["lr"]
            training_result.append(
                [
                    t,
                    actual_lr,
                    wt,
                    train_loss,
                    test_loss,
                    train_r2,
                    test_r2,
                    duration,
                ]
            )
            torch.cuda.empty_cache()

            # time consumption and report R-squared values.
            print(
                f"Epoch {t+1}: Lr: {actual_lr:.3e}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, "
                + "Train_r2: {train_r2:.3f}, Test_r2: {test_r2:.3f}, Duration(min):  {duration:.2f}"
            )

            if best_loss > test_loss:
                best_loss = test_loss
                best_epoch = t + 1
                
                model_save_path = os.path.join(
                    save_dir,
                    "model", 
                    f"{config['model']['name']}_{year}{month:02d}_reg_{config['model']['mode']}.pth"
                    )
                # save model
                torch.save(
                    {
                        "epoch": t,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "MAEloss": test_loss,
                        "R-squared": test_r2
                    },
                    model_save_path,
                )

                # save prediction array
                pred_path = os.path.join(
                    save_dir,
                    "log",
                    f"{config['model']['name']}_{year}{month:02d}_reg_{config['model']['mode']}.npz"
                )

                np.savez(pred_path, train=train_dict, test=test_dict)


    training_result.append(
        [
            f"Hyper parameters: batch_size: {config['optimize']['batch_size']}, "
            + f"number of epoch: {config['scheduler']['epochs']}"
        ]
    )

    # save the results
    # print("Saving the model's result")
    df_result = pd.DataFrame(
        training_result,
        columns=[
            "Epoch",
            "learning rate",
            "weight decay",
            "Train_loss",
            "Test_loss",
            "train_r2",
            "test_r2",
            "Training-testing time(min)",
        ],
    )

    total_save_path = os.path.join(
        save_dir,
        "optimization",
        f"{config['model']['name']}_{year}{month:02d}_validation_{config['model']['mode']}_results.csv"
    )

    print("Save file here:", total_save_path)
    df_result.to_csv(total_save_path, index=False)

    print("Done!")
