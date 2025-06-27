import torch
import torch.nn as nn


def init_weights(module: nn.Module):
    """Applies Xavier initialization to Conv/Linear layers."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ConvNormBN(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        padding: int = None,
    ):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x


class LinearNorm(nn.Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.apply(init_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)