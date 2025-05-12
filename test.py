import torch
import numpy as np

if __name__ == "__main__":

    # result = np.load("./results/uncertainty/Resnet50_cp_AllRegions.npz", allow_pickle=True)
    # print(result.files)
    # print(result['cp'][0][0])
    # print(result['val'][0])
    # print(result['label'][0])

    result = np.load("./results/log/Resnet18_202504_reg_qr_qlow2_qhigh98.npz", allow_pickle=True)
    
    pred_train = result['train'].item()['prediction']
    label_train = result['train'].item()['label']
    pred = result['test'].item()['prediction']
    label = result['test'].item()['label']

    check_train = np.maximum(pred_train[:, 0] - label_train[:, 0], label_train[:, 0] - pred_train[:, 1])
    check = np.maximum(pred[:, 0] - label[:, 0], label[:, 0] - pred[:, 1])
    print(np.quantile(check_train, q=0.95, method='higher'))
    print(np.quantile(check, q=0.95, method='higher'))
    

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:0" if use_cuda else "cpu")
    
    model_file = "./results/model/Resnet18_202504_reg_qr_qlow2_qhigh98.pth"
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    print(checkpoint.keys())
    print(checkpoint['epoch'])
    print(checkpoint['MAEloss'])
    print(checkpoint['R-squared'])