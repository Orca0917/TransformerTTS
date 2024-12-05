import torch
import torch.nn as nn
from model.layers import LinearNorm
from model.module import (
    EncoderPreNet, 
    DecoderPreNet, 
    ScaledPositionalEncoding, 
    Transformer,
    PostNet
)


class TransformerTTS(nn.Module):
    def __init__(self, m_config):
        super(TransformerTTS, self).__init__()
        # configuration
        d_hidden    = m_config["common"]["d_hidden"]
        self.n_mels = m_config["common"]["n_mels"]

        self.encoder_prenet = EncoderPreNet(m_config)
        self.decoder_prenet = DecoderPreNet(m_config)
        self.scaled_positional_encoding = ScaledPositionalEncoding(m_config)
        self.transformer = Transformer(m_config)
        self.mel_linear = LinearNorm(d_hidden, self.n_mels)
        self.stop_linear = LinearNorm(d_hidden, 1)
        self.postnet = PostNet(m_config)
    

    def forward(self, phoneme, melspectrogram, src_len, mel_len, **kwargs):
        """
        Forward pass for TransformerTTS
        
        Args:
            phoneme: [batch_size, max_src_len]
            melspectrogram: [batch_size, max_mel_len, n_mels]
            src_len: [batch_size]
            mel_len: [batch_size]
        """

        batch_size = melspectrogram.size(0)
        device = melspectrogram.device
        
        # Encoder prenet
        prenet_src_out = self.encoder_prenet(phoneme)
        src = self.scaled_positional_encoding(prenet_src_out)

        # Decoder prenet
        go_frame = torch.zeros(batch_size, 1, self.n_mels).to(device)
        mel_input = torch.cat([go_frame, melspectrogram[:, :-1, :]], dim=1)
        prenet_tgt_out = self.decoder_prenet(mel_input)
        tgt = self.scaled_positional_encoding(prenet_tgt_out)
        
        # Transformer
        transformer_out = self.transformer(src, tgt, src_len, mel_len)

        # Output layers
        mel_output = self.mel_linear(transformer_out)
        stop_token = self.stop_linear(transformer_out).squeeze(-1)
        mel_output_postnet = self.postnet(mel_output) + mel_output
        
        return {
            "mel_output": mel_output,
            "stop_token": stop_token,
            "mel_output_postnet": mel_output_postnet
        }
    

    @torch.no_grad()
    def inference(self, phoneme, src_len, max_decoder_steps=1000, **kwargs):
        self.eval()  # 추론 모드로 전환하여 dropout 등 비활성화

        batch_size = phoneme.size(0)
        device = phoneme.device

        # Encoder
        prenet_src_out = self.encoder_prenet(phoneme)  # [batch_size, src_len, d_model]
        src = self.scaled_positional_encoding(prenet_src_out)
        src_key_padding_mask = self.transformer._get_key_padding_masks(src_len)
        memory = self.transformer.transformer.encoder(
            src=src,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        )

        # Decoder 초기화
        ys = torch.zeros(batch_size, 1, self.n_mels).to(device)
        decoded_mel = []
        stop_tokens = []
        not_finished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_decoder_steps):
            # Decoder PreNet 및 Positional Encoding
            prenet_tgt_out = self.decoder_prenet(ys)
            tgt = self.scaled_positional_encoding(prenet_tgt_out)

            # Transformer Decoder 호출
            tgt_mask = self.transformer.transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            transformer_out = self.transformer.transformer.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            # 마지막 타임스텝의 출력만 사용
            frame_pred = transformer_out[:, -1, :]
            mel_output = self.mel_linear(frame_pred)
            stop_token = self.stop_linear(frame_pred).squeeze(-1)

            decoded_mel.append(mel_output.unsqueeze(1))
            stop_tokens.append(stop_token.unsqueeze(1))

            # Stop Token 확인
            stop_probs = torch.sigmoid(stop_token)  # [batch_size]
            not_finished = not_finished & (stop_probs < 0.5)
            if not not_finished.any():
                break

            # 다음 입력 준비
            mel_output = mel_output * not_finished.unsqueeze(1)  # 완료된 배치는 업데이트하지 않음
            ys = torch.cat([ys, mel_output.unsqueeze(1)], dim=1)  # [batch_size, T+1, n_mels]

        # 출력들을 하나의 텐서로 결합
        decoded_mel = torch.cat(decoded_mel, dim=1)
        stop_tokens = torch.cat(stop_tokens, dim=1)

        # PostNet 적용
        mel_output_postnet = self.postnet(decoded_mel) + decoded_mel  # [batch_size, total_steps, n_mels]

        return {
            "mel_output": decoded_mel, 
            "stop_token": stop_tokens, 
            "mel_output_postnet": mel_output_postnet
        }
