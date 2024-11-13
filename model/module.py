import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class EncoderPreNet(nn.Module):
    def __init__(self, m_config):
        super(EncoderPreNet, self).__init__()
        
        # Load configuration parameters
        n_phoneme = m_config["encoder"]["prenet"]["n_phonemes"]
        d_hidden = m_config["encoder"]["prenet"]["d_hidden"]
        n_convolutions = m_config["encoder"]["prenet"]["n_convolutions"]
        kernel_size = m_config["encoder"]["prenet"]["kernel_size"]

        # Embedding layer for phonemes
        self.embedding = nn.Embedding(n_phoneme, d_hidden)
        
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
        in_dim = m_config["decoder"]["prenet"]["in_dim"]
        sizes = m_config["decoder"]["prenet"]["sizes"]
        
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
        n_mel_channels = m_config["common"]["n_mels"]
        postnet_embedding_dim = m_config["postnet"]["embedding_dim"]
        postnet_kernel_size = m_config["postnet"]["kernel_size"]
        postnet_n_convolutions = m_config["postnet"]["n_convolutions"]
        
        self.convolutions = nn.ModuleList()
        
        # Initial convolution layer
        self.convolutions.append(nn.Sequential(
            ConvNorm(n_mel_channels, postnet_embedding_dim,
                     kernel_size=postnet_kernel_size, stride=1,
                     padding=(postnet_kernel_size - 1) // 2, w_init_gain='tanh'),
            nn.BatchNorm1d(postnet_embedding_dim)
        ))
        
        # Intermediate convolution layers
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(nn.Sequential(
                ConvNorm(postnet_embedding_dim, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=(postnet_kernel_size - 1) // 2, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim)
            ))
        
        # Final convolution layer
        self.convolutions.append(nn.Sequential(
            ConvNorm(postnet_embedding_dim, n_mel_channels,
                     kernel_size=postnet_kernel_size, stride=1,
                     padding=(postnet_kernel_size - 1) // 2, w_init_gain='linear'),
            nn.BatchNorm1d(n_mel_channels)
        ))

    def forward(self, x):
        
        x = x.transpose(1, 2)
        for conv in self.convolutions[:-1]:
            x = F.dropout(torch.tanh(conv(x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x.transpose(1, 2)
    
    
class ScaledPositionalEncoding(nn.Module):
    def __init__(self, m_config):
        super(ScaledPositionalEncoding, self).__init__()
        
        # Load configuration parameters
        max_len = m_config["positional_encoding"]["max_seq_len"]
        d_hidden = m_config["positional_encoding"]["d_hidden"]
        
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
    

class Encoder(nn.Module):
    def __init__(self, m_config):
        super(Encoder, self).__init__()
        
        # Load configuration parameters
        d_model = m_config["encoder"]["transformer"]["d_hidden"]
        n_head = m_config["encoder"]["transformer"]["n_heads"]
        d_feedforward = m_config["encoder"]["transformer"]["d_feedforward"]
        n_layers = m_config["encoder"]["transformer"]["n_layers"]
        dropout = m_config["encoder"]["transformer"]["dropout"]
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=n_layers
        )
        
    def forward(self, encoder_pe_out, mask, src_key_padding_mask):
        return self.encoder(src=encoder_pe_out, mask=mask, src_key_padding_mask=src_key_padding_mask)


class Decoder(nn.Module):
    def __init__(self, m_config):
        super(Decoder, self).__init__()
        
        # Load configuration parameters
        d_model = m_config["decoder"]["transformer"]["d_hidden"]
        n_head = m_config["decoder"]["transformer"]["n_heads"]
        d_feedforward = m_config["decoder"]["transformer"]["d_feedforward"]
        n_layers = m_config["decoder"]["transformer"]["n_layers"]
        dropout = m_config["decoder"]["transformer"]["dropout"]
        n_mels = m_config["common"]["n_mels"]
        self.n_mels = n_mels
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=n_layers
        )

        # Output layers
        self.mel_linear = LinearNorm(d_model, n_mels)
        self.stop_linear = LinearNorm(d_model, 1)
        self.postnet = PostNet(m_config)

    def forward(self, decoder_pe_out, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        decoder_output = self.decoder(
            tgt=decoder_pe_out, 
            memory=memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask
        )

        mel_output = self.mel_linear(decoder_output)
        stop_token = self.stop_linear(decoder_output)
        mel_output_postnet = self.postnet(mel_output) + mel_output
        
        return mel_output, stop_token.squeeze(2), mel_output_postnet
