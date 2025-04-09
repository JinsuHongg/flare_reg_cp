import torch
import numpy as np


def train_loop(dataloader, model, loss_fn, optimizer=None, lr_scheduler=None):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    device = next(model.parameters()).device
    model.train()

    # size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # pred_arr = np.empty((0, 6), int)
    pred_dict = {
        'prediction': np.empty((0, 1), int),
        'label': np.empty((0, 1), int)
        }
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        y = y.to(torch.float32)

        # Compute prediction and loss
        pred = model(X).squeeze(-1)
        
        # Move tensors to CPU and convert them to NumPy arrays only once
        predictions_numpy = pred.unsqueeze(1).cpu().detach().numpy()
        y_numpy = y.unsqueeze(1).cpu().detach().numpy()
        # Use a loop to avoid repeated np.concatenate calls
        data_list = [predictions_numpy, y_numpy]
        for key, data in zip(pred_dict.keys(), data_list):
            pred_dict[key] = np.concatenate((pred_dict[key], data), axis=0)

        loss = loss_fn(
            pred, y
        )  # log_softmax after NLLloss! torch.nn.functional.log_softmax(pred, dim=1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:.4f}  [{current:>5d}/{size:>5d}]")

        train_loss += loss

    # check loss
    train_loss /= num_batches
    # print(f'Training Avg loss: {train_loss:.4f}')

    return float(train_loss), pred_dict


def test_loop(dataloader, model, loss_fn, softmax = None):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    device = next(model.parameters()).device
    model.eval()
    num_batches = len(dataloader)

    test_loss = 0
    # pred_arr = np.empty((0, 6), float)
    pred_dict = {
        'prediction': np.empty((0, 1), int),
        'label': np.empty((0, 1), int)
        }
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.to(torch.float32)
            pred = model(X).squeeze(-1)

            test_loss += loss_fn(
                pred, y
            ).item()  # torch.nn.functional.log_softmax(pred, dim=1)
            
            # Move tensors to CPU and convert them to NumPy arrays only once
            predictions_numpy = pred.unsqueeze(1).cpu().detach().numpy()
            y_numpy = y.unsqueeze(1).cpu().detach().numpy()

            # Use a loop to avoid repeated np.concatenate calls
            data_list = [predictions_numpy, y_numpy]
            for key, data in zip(pred_dict.keys(), data_list):
                pred_dict[key] = np.concatenate((pred_dict[key], data), axis=0)

    test_loss /= num_batches

    # print(f"Test loss: {test_loss:.4f}")
    return test_loss, pred_dict