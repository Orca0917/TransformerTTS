import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import ConvNormBN, LinearNorm
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer
)
from model.transformer.layers import TransformerDecoderLayer, TransformerDecoder


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

        self.dropout = dropout

        # N(3)-layer ConvNormBN blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(n_layers):
            in_dim = in_channels if i == 0 else out_channels
            self.conv_blocks.append(ConvNormBN(in_dim, out_channels, kernel_size))

        # Final linear layer
        self.linear = LinearNorm(out_channels, out_channels)

    def forward(self, phoneme_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phoneme_emb (Tensor): (B, T_phon, H_emb)
        Returns:
            enc_prenet_out (Tensor): (B, T_phon, H)
        """

        for conv in self.conv_blocks:
            phoneme_emb = F.dropout(F.relu(conv(phoneme_emb)), p=self.dropout, training=self.training)

        enc_prenet_out = self.linear(phoneme_emb)
        return enc_prenet_out


class DecoderPreNet(nn.Module):
    def __init__(
        self, 
        n_mels: int, 
        d_model: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.linear1 = LinearNorm(n_mels, d_model)
        self.linear2 = LinearNorm(d_model, d_model)
        self.dropout = dropout

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.relu(self.linear1(melspec)), p=self.dropout, training=self.training)
        dec_prenet_out = self.linear2(x)
        return dec_prenet_out


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: str,
        max_len: int = 5000,
    ):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device, dtype=torch.float32)
        self.register_buffer('pe', pe)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)) / d_model)

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, H)
        Returns:
            Tensor: Positionally encoded tensor of shape (B, T, H)
        """
        x = x + self.alpha * self.pe[:x.size(1), :].unsqueeze(0)
        x = F.dropout(x, p=0.1, training=self.training)
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

        self.conv_layers = nn.ModuleList()

        # 1st layer of PostNet
        self.conv_layers.append(ConvNormBN(in_channels, out_channels, kernel_size))
        self.conv_layers.append(nn.Tanh())
        self.conv_layers.append(nn.Dropout(dropout))
        
        # 2~4th layers of PostNet
        for _ in range(n_layers - 2):
            self.conv_layers.append(ConvNormBN(out_channels, out_channels, kernel_size))
            self.conv_layers.append(nn.Tanh())
            self.conv_layers.append(nn.Dropout(dropout))

        # Last layer of PostNet
        self.conv_layers.append(ConvNormBN(out_channels, in_channels, kernel_size))
        self.conv_layers.append(nn.Dropout(dropout))


    def forward(self, pred_mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_mel (Tensor): Input tensor of shape (B, T_mel, n_mels)
        Returns:
            post_mel (Tensor): Output tensor of shape (B, T_mel, n_mels)
        """

        for layer in self.conv_layers:
            pred_mel = layer(pred_mel)

        return pred_mel


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
            activation='relu',
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


    def _get_mask(self, phoneme_lens: torch.Tensor = None, mel_lens: torch.Tensor = None):

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
        phoneme: torch.Tensor, 
        melspec: torch.Tensor, 
        phoneme_lens: torch.Tensor,
        melspec_lens: torch.Tensor, 
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

        # decoder
        tgt_in = self.pe(self.dec_prenet(tgt_in))
        tgt_out, alignments = self.decoder(
            tgt=tgt_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
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


    def inference(
        self, 
        phoneme: torch.Tensor, 
        phoneme_lens: torch.Tensor, 
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

            _, _, tgt_mask = self._get_mask(mel_lens=torch.tensor([t] * B, device=self.device))
            tgt_out, _ = self.decoder(
                tgt=tgt_in,
                memory=memory,
                tgt_mask=tgt_mask,
                # tgt_is_causal=True,
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