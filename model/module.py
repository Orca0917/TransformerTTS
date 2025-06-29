import torch.nn as nn
from torch import Tensor

class ConvNormBN(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        padding: int = None,
        activation: str = 'relu',
    ):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)

        gain = nn.init.calculate_gain(activation)
        nn.init.xavier_normal_(self.conv.weight, gain=gain)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
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
        activation: str = 'relu',
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        gain = nn.init.calculate_gain(activation)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)