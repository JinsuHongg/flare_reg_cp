import torch

# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn
from torchvision.models import (
    resnet152, resnet101, resnet50, resnet34, resnet18,
    ResNet152_Weights, ResNet101_Weights, ResNet50_Weights, ResNet18_Weights
)

def get_model(model_cfg: dict):
    name = model_cfg.get("name", "Resnet18")
    dropout = model_cfg.get("dropout", 0.5)
    mode = model_cfg.get("mode", "cp")

    if name == "Resnet18":
        return Resnet18(dropout=dropout, mode=mode)
    elif name == "Resnet34":
        return Resnet50(dropout=dropout, mode=mode)
    elif name == "Resnet50":
        return Resnet50(dropout=dropout, mode=mode)
    elif name == "Resnet101":
        return Resnet101(dropout=dropout, mode=mode)
    elif name == "Resnet152":
        return Resnet152(dropout=dropout, mode=mode)
    else:
        raise ValueError(f"Unknown model name: {name}")

class Resnet18(nn.Module):
    def __init__(self, dropout: float = 0.5, mode:str = 'CP') -> None:
        super(Resnet18, self).__init__()

        # load pretrained architecture from pytorch
        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # self.model = torch.hub.load(
        #     "pytorch/vision:v0.10.0", "resnet18", weights=ResNet18_Weights.IMAGENET1K_V1
        # )
        
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected (FC) layer with dropout
        if mode == 'cp':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        elif mode == 'qr' or mode == 'cqr':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 2),
                nn.ReLU()
            )
        elif mode == 'mcd':
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),  # Apply MC Dropout
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are: 'cp', 'qr', 'cqr', 'mcd'.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty

class Resnet34(nn.Module):
    def __init__(self, dropout: float = 0.5, mode:str = 'CP') -> None:
        super(Resnet34, self).__init__()

        # load pretrained architecture from pytorch
        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # self.model = torch.hub.load(
        #     "pytorch/vision:v0.10.0", "resnet18", weights=ResNet18_Weights.IMAGENET1K_V1
        # )
        
        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected (FC) layer with dropout
        if mode == 'cp':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        elif mode == 'qr' or mode == 'cqr':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 2),
                nn.ReLU()
            )
        elif mode == 'mcd':
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),  # Apply MC Dropout
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are: 'cp', 'qr', 'cqr', 'mcd'.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty

class Resnet50(nn.Module):
    def __init__(self, dropout: float = 0.5, mode:str = "cp") -> None:
        super(Resnet50, self).__init__()

        # load pretrained architecture from pytorch
        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # self.model = torch.hub.load(
        #     "pytorch/vision:v0.10.0", "resnet50", weights=ResNet50_Weights.IMAGENET1K_V1
        # )

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected (FC) layer with dropout
        if mode == 'cp':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        elif mode == 'qr' or mode == 'cqr':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 2),
                nn.ReLU()
            )
        elif mode == 'mcd':
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),  # Apply MC Dropout
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are: 'cp', 'qr', 'cqr', 'mcd'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty
    
class Resnet101(nn.Module):
    def __init__(self, dropout: float = 0.5, mode:str = "cp") -> None:
        super(Resnet101, self).__init__()

        # load pretrained architecture from pytorch
        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # self.model = torch.hub.load(
        #     "pytorch/vision:v0.10.0", "resnet50", weights=ResNet50_Weights.IMAGENET1K_V1
        # )

        self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected (FC) layer with dropout
        if mode == 'cp':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        elif mode == 'qr' or mode == 'cqr':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 2),
                nn.ReLU()
            )
        elif mode == 'mcd':
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),  # Apply MC Dropout
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are: 'cp', 'qr', 'cqr', 'mcd'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty
    
class Resnet152(nn.Module):
    def __init__(self, dropout: float = 0.5, mode:str = "cp") -> None:
        super(Resnet152, self).__init__()

        # load pretrained architecture from pytorch
        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # self.model = torch.hub.load(
        #     "pytorch/vision:v0.10.0", "resnet50", weights=ResNet50_Weights.IMAGENET1K_V1
        # )

        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected (FC) layer with dropout
        if mode == 'cp':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        elif mode == 'qr' or mode == 'cqr':
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 2),
                nn.ReLU()
            )
        elif mode == 'mcd':
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),  # Apply MC Dropout
                nn.Linear(self.model.fc.in_features, 1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are: 'cp', 'qr', 'cqr', 'mcd'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty
