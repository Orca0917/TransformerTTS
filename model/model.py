# https://github.com/choiHkk/Transformer-TTS/blob/main/model.py

from torch.autograd import Variable
from .module import (
    TextPrenet, 
    PositionalEncoding, 
    Prenet, 
    Encoder, 
    EncoderLayer, 
    Decoder, 
    DecoderLayer, 
    LinearNorm, 
    Postnet
)
import torch.nn as nn
import torch
import math

class TransformerTTS(nn.Module):
    def __init__(self, hparams):
        super(TransformerTTS, self).__init__()
        n_symbols = hparams.n_symbols
        n_speakers = hparams.n_speakers
        d_model = hparams.d_model
        n_mel_channels = hparams.n_mel_channels
        self.n_speakers = n_speakers
        self.n_mel_channels = n_mel_channels
        
        self.embedding = nn.Embedding(n_symbols, d_model*2)
        std = math.sqrt(2.0 / (n_symbols + d_model))
        val = math.sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.text_prenet = TextPrenet(encoder_embedding_dim=d_model)
        self.positinoal_encoding = PositionalEncoding(d_model)
        self.prenet = Prenet(in_dim=n_mel_channels, sizes=[d_model,d_model])
        self.encoder = Encoder(
            encoder_layer=EncoderLayer(d_model=d_model), 
            norm=nn.LayerNorm(normalized_shape=d_model))
        self.decoder = Decoder(
            decoder_layer=DecoderLayer(d_model=d_model), 
            norm=nn.LayerNorm(normalized_shape=d_model))
        self.linear_projection = LinearNorm(d_model, n_mel_channels)
        self.postnet = Postnet(n_mel_channels=n_mel_channels)
        self.gate_projection = LinearNorm(d_model, 1, w_init_gain='sigmoid')
        if n_speakers > 0:
            self.speaker_embedding = nn.Embedding(n_speakers, d_model)
            std = math.sqrt(2.0 / (n_speakers + d_model))
            val = math.sqrt(3.0) * std  # uniform bounds for std
            self.speaker_embedding.weight.data.uniform_(-val, val)
        
    def generate_square_subsequent_mask(self, lsz, rsz):
        """https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer"""
        return torch.triu(torch.ones(lsz, rsz) * float('-inf'), diagonal=1)
    
    def generate_padding_mask(self, lengths, max_len=None):
        """https://github.com/ming024/FastSpeech2/blob/master/utils/tools.py"""
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(
            dtype=lengths.dtype, device=lengths.device)
        return ids >= lengths.unsqueeze(1).expand(-1, max_len)
    
    def initialize_masks(self, x_lengths=None, y_lengths=None):
        """
        - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
          `(N, S, E)` if `batch_first=True`.
        - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
          `(N, T, E)` if `batch_first=True`.
        - src_mask: :math:`(S, S)`.
        - tgt_mask: :math:`(T, T)`.
        - memory_mask: :math:`(T, S)`.
        - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
        - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
        - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
        """
        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None
        self.src_key_padding_mask = None
        self.tgt_key_padding_mask = None
        if x_lengths is not None:
            S = x_lengths.max().item()
            self.src_mask = self.generate_square_subsequent_mask(S, S).to(device=x_lengths.device)         # text sequence self-attention mask
            self.src_key_padding_mask = self.generate_padding_mask(x_lengths).to(device=x_lengths.device)  # text sequence padding mask
        if y_lengths is not None:
            T = y_lengths.max().item()
            self.tgt_mask = self.generate_square_subsequent_mask(T, T).to(device=y_lengths.device)         # mel sequence self-attention mask
            self.tgt_key_padding_mask = self.generate_padding_mask(y_lengths).to(device=y_lengths.device)  # mel sequence padding mask
        if x_lengths is not None and y_lengths is not None:
            T = y_lengths.max().item()
            S = x_lengths.max().item()
            self.memory_mask = self.generate_square_subsequent_mask(T, S).to(device=y_lengths.device)      # text-mel cross attention mask
            
    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.n_mel_channels).zero_())
        return decoder_input
    
    def to_gpu(self, x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
        return torch.autograd.Variable(x)
    
    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = self.to_gpu(text_padded).long()
        input_lengths = self.to_gpu(input_lengths).long()
        mel_padded = self.to_gpu(mel_padded).float()
        gate_padded = self.to_gpu(gate_padded).float()
        output_lengths = self.to_gpu(output_lengths).long()
        return (
            (text_padded, input_lengths, mel_padded, output_lengths),
            (mel_padded.clone(), gate_padded))
            
    def parse_output(self, outputs):
        outputs[0].data.masked_fill_(
            self.tgt_key_padding_mask.unsqueeze(-1).repeat(1,1,outputs[0].size(-1)), 0.0)
        outputs[1].data.masked_fill_(
            self.tgt_key_padding_mask.unsqueeze(-1).repeat(1,1,outputs[0].size(-1)), 0.0)
        outputs[2].data.masked_fill_(
            self.tgt_key_padding_mask.unsqueeze(-1), 1e3)  # gate energies
        return outputs
        
    def forward(self, x, x_lengths=None, y=None, y_lengths=None, speakers=None):
        self.train()
        self.initialize_masks(x_lengths=x_lengths, y_lengths=y_lengths)
        if speakers is None:
            speakers = torch.LongTensor([0]).repeat(x.size(0)).to(x.device)
        x = self.text_prenet(self.embedding(x))
        x = self.positinoal_encoding(x)
        memory = self.encoder(
            src=x, 
            mask=self.src_mask, 
            src_key_padding_mask=self.src_key_padding_mask
        )
        if self.n_speakers > 0:
            assert speakers is not None
            g = self.speaker_embedding(speakers).unsqueeze(1).repeat(1,x_lengths.max().item(),1)
            memory = memory + g
        y = torch.cat([self.get_go_frame(memory).unsqueeze(1), y[:,:-1,:]], dim=1)
        y = self.positinoal_encoding(self.prenet(y))
        features = self.decoder(
            tgt=y, 
            memory=memory, 
            tgt_mask=self.tgt_mask, 
            memory_mask=self.memory_mask, 
            tgt_key_padding_mask=self.tgt_key_padding_mask, 
            memory_key_padding_mask=self.src_key_padding_mask
        )
        mel = self.linear_projection(features)
        gate = self.gate_projection(features)
        post_mel = self.postnet(mel.transpose(1,2)).transpose(1,2) + mel
        return self.parse_output([post_mel, mel, gate])
    
    @torch.no_grad()
    def inference(self, x, x_lengths=None, speakers=None, gate_threshold=0.5, max_len=1000):
        self.eval()
        self.initialize_masks(x_lengths=x_lengths)
        if speakers is None:
            speakers = torch.LongTensor([0]).repeat(x.size(0)).to(x.device)
        x = self.text_prenet(self.embedding(x))
        x = self.positinoal_encoding(x)
        memory = self.encoder(
            src=x, 
            mask=self.src_mask, 
            src_key_padding_mask=self.src_key_padding_mask
        )
        if self.n_speakers > 0:
            assert speakers is not None
            g = self.speaker_embedding(speakers).unsqueeze(1).repeat(1,x_lengths.max().item(),1)
            memory = memory + g
        go_frame = self.get_go_frame(memory).unsqueeze(1)
        y_hat = torch.FloatTensor([]).to(x.device)
        gate_outputs = torch.FloatTensor([]).to(x.device)
        y_hat = torch.cat([y_hat, go_frame], dim=1)
        while True:
            y_lengths = torch.LongTensor([y_hat.size(1)]).to(x.device)
            self.initialize_masks(x_lengths=x_lengths, y_lengths=y_lengths)
            features = self.decoder(
                tgt=self.positinoal_encoding(self.prenet(y_hat)), 
                memory=memory, 
                tgt_mask=self.tgt_mask, 
                memory_mask=self.memory_mask, 
                tgt_key_padding_mask=self.tgt_key_padding_mask, 
                memory_key_padding_mask=self.src_key_padding_mask
            )
            frame = self.linear_projection(features)
            gate = self.gate_projection(features)
            if torch.sigmoid(gate[:,-1].data) > gate_threshold:
                break
            elif y_hat.size(1) == max_len:
                print("Warning! Reached max decoder steps")
                break
            else:
                y_hat = torch.cat([y_hat, frame[:,-1:,:]], dim=1)
                gate_outputs = torch.cat([gate_outputs, gate[:,-1:,:]], dim=1)
        post_y_hat = self.postnet(y_hat.transpose(1,2)).transpose(1,2) + y_hat
        return self.parse_output([post_y_hat, y_hat, gate])
    