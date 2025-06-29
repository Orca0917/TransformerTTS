import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .module import ConvNormBN, LinearNorm
from .layers import TransformerDecoderLayer, TransformerDecoder


class EncoderPreNet(nn.Module):
    """
    Encoder Pre-Net: N-layer ConvNormBN blocks followed by a final LinearNorm layer.
    Input/Output shape: (B, T, H)
    """

    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        # N(3)-layer ConvNormBN blocks
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = in_channels if i == 0 else out_channels
            self.layers.append(ConvNormBN(in_dim, out_channels, kernel_size))
            self.layers.append(nn.Dropout(dropout))

        # Final linear layer
        self.linear = LinearNorm(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        '''
        x       : (#bs, #phon, #hidden)
        '''
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)
        return x


class DecoderPreNet(nn.Module):
    def __init__(
        self, 
        n_mels: int, 
        d_model: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.linear1 = LinearNorm(n_mels, d_model, activation='relu')
        self.linear2 = LinearNorm(d_model, d_model, activation='relu')
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        '''
        x       : (#bs, #frame, #mel)
        '''
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout2(F.relu(self.linear2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: str,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)) / d_model)

        pe = torch.zeros(max_len, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        '''
        x       :(#bs, #frame, #hidden) or (#bs, #phon, #hidden)
        '''
        x = x + self.alpha * self.pe[:x.size(1), :].unsqueeze(0)
        x = self.dropout(x)
        return x


class PostNet(nn.Module):
    def __init__(
        self, 
        n_layers: int, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.5,
    ):
        super(PostNet, self).__init__()

        self.layers = nn.ModuleList()

        # 1st layer of PostNet
        self.layers.append(ConvNormBN(in_channels, out_channels, kernel_size, activation='tanh'))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Dropout(dropout))
        
        # 2~4th layers of PostNet
        for _ in range(n_layers - 2):
            self.layers.append(ConvNormBN(out_channels, out_channels, kernel_size, activation='tanh'))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(dropout))

        # Last layer of PostNet
        self.layers.append(ConvNormBN(out_channels, in_channels, kernel_size, activation='tanh'))
        self.layers.append(nn.Dropout(dropout))


    def forward(self, x: Tensor) -> Tensor:
        '''
        x       : (#bs, #frame, #mel)
        '''
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerTTS(nn.Module):
    def __init__(
        self, 
        encoder_prenet_n_layers: int, 
        encoder_prenet_in_channel: int,
        encoder_prenet_out_channel: int,
        encoder_prenet_kernel_size: int,
        encoder_prenet_dropout: float,
        encoder_n_layers: int,
        encoder_n_head: int,
        encoder_d_ffn: int,
        encoder_dropout: float,
        decoder_n_layers: int,
        decoder_n_head: int,
        decoder_d_ffn: int,
        decoder_dropout: float,
        postnet_n_layers: int,
        postnet_kernel_size: int,
        postnet_dropout: float,
        d_model: int,
        n_phon: int = 100,
        n_mels: int = 80,
        device: str = 'cuda',
    ):
        super().__init__()

        self.device = device
        self.n_mels = n_mels

        # embedding
        self.emb = nn.Embedding(n_phon, d_model)
        
        # prenet
        self.enc_prenet = EncoderPreNet(
            encoder_prenet_n_layers,
            encoder_prenet_in_channel,
            encoder_prenet_out_channel,
            encoder_prenet_kernel_size,
            encoder_prenet_dropout,
        )

        self.dec_prenet = DecoderPreNet(
            n_mels, d_model
        )

        # positional encoding
        self.pe = PositionalEncoding(
            d_model, device
        )

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_n_head,
            dim_feedforward=encoder_d_ffn,
            dropout=encoder_dropout,
            activation='relu',
            batch_first=True,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=encoder_n_layers
        )

        # transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_n_head,
            dim_feedforward=decoder_d_ffn, 
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=decoder_n_layers
        )

        # postnet
        self.postnet = PostNet(
            postnet_n_layers,
            n_mels,
            d_model,
            postnet_kernel_size,
            postnet_dropout,
        )

        # linears
        self.linear1 = LinearNorm(d_model, n_mels)
        self.linear2 = LinearNorm(d_model, 1)  # Stop token prediction


    def _get_mask(self, phoneme_lens: Tensor = None, mel_lens: Tensor = None):

        src_key_padding_mask = None
        tgt_key_padding_mask = None
        tgt_mask = None
        
        if phoneme_lens is not None:
            max_phon_len = phoneme_lens.max().item()
            src_key_padding_mask = (
                torch.arange(max_phon_len, device=self.device)
                .unsqueeze(0)
                .expand(phoneme_lens.size(0), max_phon_len)
                >= phoneme_lens.unsqueeze(1)
            )

        if mel_lens is not None:
            max_mel_len = mel_lens.max().item()
            tgt_key_padding_mask = (
                torch.arange(max_mel_len, device=self.device)
                .unsqueeze(0)
                .expand(mel_lens.size(0), max_mel_len)
                >= mel_lens.unsqueeze(1)
            )
            tgt_mask = torch.triu(
                torch.ones(max_mel_len, max_mel_len, device=self.device), 
                diagonal=1
            ).bool()

        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask


    def forward(
        self, 
        phoneme: Tensor, 
        melspec: Tensor, 
        phoneme_lens: Tensor,
        melspec_lens: Tensor, 
    ) -> dict:
        """
        Args:
          - phoneme_input (Tensor): Input phoneme tensor (B, T_phon)
          - melspec (Tensor): Ground-truth mel spectrogram (B, T_mel, n_mels)
          - phoneme_lens (Tensor): Lengths of each phoneme sequence (B,)
          - melspec_lens (Tensor): Lengths of each mel sequence (B,)

        Returns:
          - Dict[str, Tensor]
        """

        go_frame = torch.zeros_like(melspec[:, :1, :])
        tgt_in = torch.cat((go_frame, melspec[:, :-1, :]), dim=1)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            tgt_mask
        ) = self._get_mask(phoneme_lens, melspec_lens)

        # encoder
        src_in = self.pe(self.enc_prenet(self.emb(phoneme)))
        memory = self.encoder(
            src=src_in,
            src_key_padding_mask=src_key_padding_mask
        )

        assert src_key_padding_mask.dim() == 2 and src_key_padding_mask.size(1) == memory.size(1)
        
        # decoder
        tgt_in = self.pe(self.dec_prenet(tgt_in))
        tgt_out, alignments = self.decoder(
            tgt=tgt_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_is_causal=True,
            memory_is_causal=False
        )

        # mel linear and postnet
        pred_melspec = self.linear1(tgt_out)
        post_melspec = self.postnet(pred_melspec) + pred_melspec

        # stop token prediction
        pred_stop = self.linear2(tgt_out).squeeze(-1)

        return {
            'pred_melspec': pred_melspec,
            'post_melspec': post_melspec,
            'pred_stop': pred_stop,
            'alignments': alignments,
        }


    @torch.no_grad()
    def inference(
        self, 
        phoneme: Tensor, 
        phoneme_lens: Tensor, 
        max_len: int = 1500, 
        stop_threshold: float = 0.5
    ) -> dict:
        """
        Args:
            phoneme (Tensor): Input phoneme tensor (1, T_phon, H)
            max_len (int): Maximum length of the output mel spectrogram
            stop_threshold (float): Threshold for stop token prediction

        Returns:
            Tensor: Predicted mel spectrogram (1, T_mel, n_mels)
        """

        B = phoneme.size(0)
        self.eval()

        # encoder part
        src_in = self.pe(self.enc_prenet(self.emb(phoneme)))
        memory = self.encoder(
            src=src_in
        )

        # decoder
        ys = [torch.zeros(B, 1, self.n_mels, device=self.device)]
        stop_history = []

        for t in range(1, max_len):
            tgt_in = torch.cat(ys, dim=1)
            tgt_in = self.pe(self.dec_prenet(tgt_in))

            (
                src_key_padding_mask,
                tgt_key_padding_mask,
                tgt_mask
            ) = self._get_mask(phoneme_lens, torch.tensor([t] * B, device=self.device))

            assert src_key_padding_mask.dim() == 2 and src_key_padding_mask.size(1) == memory.size(1)

            tgt_out, _ = self.decoder(
                tgt=tgt_in,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_is_causal=True,
                memory_is_causal=False
            )

            cur_frame = tgt_out[:, -1:, :]
            pred_mel  = self.linear1(cur_frame)
            pred_stop = self.linear2(cur_frame).squeeze(-1)
            ys.append(pred_mel)
            stop_history.append(pred_stop)

            prob = torch.sigmoid(pred_stop)  # (B,)
            if (prob >= stop_threshold).all():
                break

        # Concatenate all predicted mel frames        
        pred_melspec = torch.cat(ys[1:], dim=1)  # Skip the initial zero frame
        post_melspec = self.postnet(pred_melspec) + pred_melspec

        return {
            'pred_melspec': pred_melspec,
            'post_melspec': post_melspec,
            'pred_stop': torch.stack(stop_history, dim=1),
        }