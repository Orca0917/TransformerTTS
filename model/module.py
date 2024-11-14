import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import LinearNorm, ConvNorm


class EncoderPreNet(nn.Module):
    def __init__(self, m_config):
        super(EncoderPreNet, self).__init__()
        
        # Load configuration parameters
        d_hidden        = m_config["common"]["d_hidden"]
        n_phonemes      = m_config["prenet"]["encoder"]["n_phonemes"]
        n_convolutions  = m_config["prenet"]["encoder"]["n_convolutions"]
        kernel_size     = m_config["prenet"]["encoder"]["kernel_size"]

        # Embedding layer for phonemes
        self.embedding = nn.Embedding(n_phonemes, d_hidden)
        
        # Convolution layers
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(d_hidden, d_hidden, kernel_size=kernel_size, 
                         stride=1, padding=(kernel_size - 1) // 2, w_init_gain='relu'),
                nn.BatchNorm1d(d_hidden)
            ) for _ in range(n_convolutions)
        ])
        
        # Final linear projection layer
        self.linear = LinearNorm(d_hidden, d_hidden)

    def forward(self, phoneme):
        x = self.embedding(phoneme)
        x = x.transpose(1, 2)
        for conv_layer in self.convolutions:
            x = F.dropout(F.relu(conv_layer(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        
        return self.linear(x)


class DecoderPreNet(nn.Module):
    def __init__(self, m_config):
        super(DecoderPreNet, self).__init__()
        
        # Load configuration parameters
        in_dim  = m_config["common"]["n_mels"]
        sizes   = m_config["prenet"]["decoder"]["sizes"]
        
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            LinearNorm(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
        return x
    
    
class PostNet(nn.Module):
    def __init__(self, m_config):
        super(PostNet, self).__init__()
        
        # Load configuration parameters
        n_mels          = m_config["common"]["n_mels"]
        d_hidden        = m_config["postnet"]["d_hidden"]
        kernel_size     = m_config["postnet"]["kernel_size"]
        n_convolutions  = m_config["postnet"]["n_convolutions"]
        
        self.convolutions = nn.ModuleList()
        
        # Initial convolution layer
        self.convolutions.append(nn.Sequential(
            ConvNorm(n_mels, d_hidden, kernel_size, w_init_gain='tanh'),
            nn.BatchNorm1d(d_hidden)
        ))
        
        # Intermediate convolution layers
        for _ in range(1, n_convolutions - 1):
            self.convolutions.append(nn.Sequential(
                ConvNorm(d_hidden, d_hidden, kernel_size, w_init_gain='tanh'),
                nn.BatchNorm1d(d_hidden)
            ))
        
        # Final convolution layer
        self.convolutions.append(nn.Sequential(
            ConvNorm(d_hidden, n_mels, kernel_size, w_init_gain='linear'),
            nn.BatchNorm1d(n_mels)
        ))

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convolutions[:-1]:
            x = F.dropout(torch.tanh(conv(x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x.transpose(1, 2)
    
    
class ScaledPositionalEncoding(nn.Module):
    def __init__(self, m_config, max_len=1024):
        super(ScaledPositionalEncoding, self).__init__()
        
        d_hidden = m_config["common"]["d_hidden"]
        
        # Positional encoding matrix initialization
        pe = torch.zeros(max_len, d_hidden)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hidden, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_hidden))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # Learnable scaling factor
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, prenet_out):
        return prenet_out + self.alpha * self.pe[:, :prenet_out.size(1), :]
    

class Transformer(nn.Module):
    def __init__(self, m_config):
        super(Transformer, self).__init__()
        d_model = m_config["transformer"]["d_hidden"]
        n_heads = m_config["transformer"]["n_heads"]
        n_encoder_layers = m_config["transformer"]["n_encoder_layers"]
        n_decoder_layers = m_config["transformer"]["n_decoder_layers"]
        d_feedforward = m_config["transformer"]["d_feedforward"]
        dropout = m_config["transformer"]["dropout"]
        self.transformer = nn.Transformer(d_model, n_heads, n_encoder_layers, n_decoder_layers, d_feedforward, dropout, batch_first=True)

    def forward(self, src, tgt, src_len, tgt_len):
        # src: (B, src_len, d_hidden)
        # tgt: (B, tgt_len, d_hidden)
        # src_len: (B, 1)
        # tgt_len: (B, 1)
        masks = self.get_masks(src_len, tgt_len)
        return self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=masks["tgt_mask"],
            src_key_padding_mask=masks["src_key_padding_mask"],
            tgt_key_padding_mask=masks["tgt_key_padding_mask"],
            memory_key_padding_mask=masks["src_key_padding_mask"],
        )

    def _get_key_padding_masks(self, lens):
        batch_size = lens.size(0)
        device = lens.device

        max_len = torch.max(lens).item()
        key_padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for idx, len in enumerate(lens):
            key_padding_mask[idx, len.item():] = True
        
        return key_padding_mask.to(device)

    def get_masks(self, src_len, tgt_len):
        # src_len: B, 1
        # tgt_len: B, 1
        device = src_len.device
        tgt_max_len = torch.max(tgt_len).item()

        src_key_padding_mask = self._get_key_padding_masks(src_len)
        tgt_key_padding_mask = self._get_key_padding_masks(tgt_len)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_max_len)

        return {
            "tgt_mask": tgt_mask.to(device),
            "src_key_padding_mask": src_key_padding_mask.to(device),
            "tgt_key_padding_mask": tgt_key_padding_mask.to(device),
        }