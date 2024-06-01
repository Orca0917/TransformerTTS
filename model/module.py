# https://github.com/choiHkk/Transformer-TTS/blob/main/modules.py

import torch.nn.functional as F
import torch.nn as nn
import torch
import math


class LinearNorm(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/layers.py"""
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/layers.py"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

    
class TextPrenet(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/model.py"""
    def __init__(self, encoder_n_convolutions=3, encoder_embedding_dim=256, encoder_kernel_size=3):
        super(TextPrenet, self).__init__()
        convolutions = []
        self.preprojection = LinearNorm(encoder_embedding_dim*2, encoder_embedding_dim*2, w_init_gain='relu')
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim*2,
                         encoder_embedding_dim*2,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim*2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.postprojection = LinearNorm(encoder_embedding_dim*2, encoder_embedding_dim)
            
    def forward(self, x, g=None):
        x = self.preprojection(x).transpose(1,2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = self.postprojection(x.transpose(1,2))
        return x
            

class Prenet(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/model.py"""
    def __init__(self, in_dim=80, sizes=[256,256]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
    
    
class Postnet(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/model.py"""
    """Postnet
        - Five 1-d convolution with 1024 channels and kernel size 5
    """

    def __init__(self, n_mel_channels=80, postnet_embedding_dim=1024, postnet_kernel_size=5, postnet_n_convolutions=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x
    
    
class PositionalEncoding(nn.Module):
    """https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0,1))
    

class EncoderLayer(nn.TransformerEncoderLayer):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html"""
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1, activation="relu", 
                 layer_norm_eps=1e-05, batch_first=True, norm_first=False, device=None, dtype=None):
        super(EncoderLayer, self).__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, 
            layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype)
        
        
class DecoderLayer(nn.TransformerDecoderLayer):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html"""
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1, activation="relu", 
                 layer_norm_eps=1e-05, batch_first=True, norm_first=False, device=None, dtype=None):
        super(DecoderLayer, self).__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, 
            layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype)


class Encoder(nn.TransformerEncoder):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html"""
    def __init__(self, encoder_layer, num_layers=3, norm=None):
        super(Encoder, self).__init__(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
class Decoder(nn.TransformerDecoder):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html"""
    def __init__(self, decoder_layer, num_layers=3, norm=None):
        super(Decoder, self).__init__(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)