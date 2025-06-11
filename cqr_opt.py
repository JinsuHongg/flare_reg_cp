# basic package
import os
# import glob
# import json
import yaml
import time
import datetime
import wandb

# import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# pytorch package
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

# predefined class
from training.loss import QuantileLoss
from training.models import get_model
from training.dataloader import  SolarFlSets
from training.trainloop import train_loop, test_loop, train_loop_qr, test_loop_qr

if __name__ == "__main__":

    # Load config file
    config_path = "./configs/config_cqr.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # initialize wandb
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="gsu-dmlab",
        # Set the wandb project where this run will be logged.
        project="uncertainty-flare-regression",
        name="CQR_optimize", # Optional: The name you want to give your run
        # Track hyperparameters and run metadata.
        config=config,
        # selec mode: online, offline
        mode = "online"
    )

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

    # print out experiment setting
    print(f"Model: {config['model']['name']}")
    print(f"Quantile: {config['model']['q_val']}")
    for key, value in config['optimize'].items():
        print(f"{key}: {value}")

    # Define dataset here!
    # train/test set
    df_train_list = [pd.read_csv(index_dir + '/' + file) for file in config['dir']['train_p']]
    df_train = pd.concat(df_train_list, ignore_index=True)
    
    df_test_list = [pd.read_csv(index_dir + '/' + file) for file in config['dir']['test_p']]
    df_test = pd.concat(df_test_list, ignore_index=True)

    # # string to datetime
    df_train["Timestamp"] = pd.to_datetime(df_train["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    df_test["Timestamp"] = pd.to_datetime(df_test["Timestamp"], format="%Y-%m-%d %H:%M:%S")

    # Define dataset
    data_train = SolarFlSets(annotations_df=df_train, img_dir=img_dir, normalization=True, target_transform=True)
    data_test = SolarFlSets(annotations_df=df_test, img_dir=img_dir, normalization=True, target_transform=True)
    print(f'Num of train: {len(data_train)}, Num of test: {len(data_test)}')

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
        
        if isinstance(config['cuda'], list):
            model = nn.DataParallel(get_model(config['model']), device_ids=config['cuda']['device']).to(device)
        else:
            model = get_model(config['model']).to(device)

        loss_fn = nn.MSELoss() if config['model']['mode'] == 'cp' else QuantileLoss(quantiles = config['model']['q_val'])
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

            train_loss, train_dict = train_loop_qr(
                train_dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=scheduler
            )
            test_loss, test_dict = test_loop_qr(
                test_dataloader, model=model, loss_fn=loss_fn
            )
            train_r2 = r2_score(train_dict['label'], np.mean(train_dict['prediction'], axis = 1)) # r2_score(y_true, y_pred)
            test_r2 = r2_score(test_dict['label'], np.mean(test_dict['prediction'], axis = 1)) # r2_score(y_true, y_pred)

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
            run.log({"Train_loss": train_loss, "Test_loss": test_loss, "train_r2": train_r2, "test_r2": test_r2})
            # time consumption and report R-squared values.
            print(
                f"Epoch {t+1}: Lr: {actual_lr:.3e}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, "
                + f"Train_r2: {train_r2:.3f}, Test_r2: {test_r2:.3f}, Duration(min):  {duration:.2f}"
            )

            if best_loss > test_loss:
                best_loss = test_loss
                best_epoch = t + 1
                
                model_save_path = os.path.join(
                    save_dir,
                    "model", 
                    (f"{config['model']['name']}_{year}{month:02d}_reg_{config['model']['mode']}_"
                    + f"qlow{config['model']['q_val'][0]*100:.0f}_qhigh{config['model']['q_val'][1]*100:.0f}.pth")
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
                    (f"{config['model']['name']}_{year}{month:02d}_reg_{config['model']['mode']}_"
                    + f"qlow{config['model']['q_val'][0]*100:.0f}_qhigh{config['model']['q_val'][1]*100:.0f}.npz")
                )

                np.savez(pred_path, train=train_dict, test=test_dict)

    # Finish the run and upload any remaining data.
    run.finish()

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
        (f"{config['model']['name']}_{year}{month:02d}_validation_{config['model']['mode']}_"
        +f"qlow{config['model']['q_val'][0]*100:.0f}_qhigh{config['model']['q_val'][1]*100:.0f}_results.csv")
    )

    print("Save file here:", total_save_path)
    df_result.to_csv(total_save_path, index=False)

    print("Done!")
