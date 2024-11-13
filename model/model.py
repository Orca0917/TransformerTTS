import torch
import torch.nn as nn
import torch.nn.functional as F
from util import create_masks, create_square_subsequent_mask
from .module import (
    EncoderPreNet, 
    DecoderPreNet, 
    ScaledPositionalEncoding, 
    Encoder, 
    Decoder
)


class TransformerTTS(nn.Module):
    def __init__(self, m_config):
        super(TransformerTTS, self).__init__()
        self.encoder_prenet = EncoderPreNet(m_config)
        self.decoder_prenet = DecoderPreNet(m_config)
        self.scaled_positional_encoding = ScaledPositionalEncoding(m_config)
        self.encoder = Encoder(m_config)
        self.decoder = Decoder(m_config)

    def forward(self, phoneme, melspectrogram, src_len, mel_len, **kwargs):
        """
        Forward pass for TransformerTTS
        
        Args:
            phoneme: [batch_size, max_src_len]
            melspectrogram: [batch_size, max_mel_len, n_mels]
            src_len: [batch_size]
            mel_len: [batch_size]
        """
        # Create all masks
        masks = create_masks(phoneme, melspectrogram, src_len, mel_len)
        
        # Encoder
        encoder_prenet_out = self.encoder_prenet(phoneme)
        encoder_pe_out = self.scaled_positional_encoding(encoder_prenet_out)
        memory = self.encoder(
            encoder_pe_out,
            mask=masks["encoder_attention_mask"],
            src_key_padding_mask=masks["src_key_padding_mask"],
        )
        
        # Decoder
        B, _, n_mels = melspectrogram.size()
        go_frame = torch.zeros(B, 1, n_mels).to(melspectrogram.device)
        melspectrogram = torch.cat([go_frame, melspectrogram[:, :-1, :]], dim=1)
        decoder_prenet_out = self.decoder_prenet(melspectrogram)
        decoder_pe_out = self.scaled_positional_encoding(decoder_prenet_out)
        decoder_output = self.decoder(
            decoder_pe_out,
            memory,
            tgt_mask=masks["tgt_mask"],
            tgt_key_padding_mask=masks["tgt_key_padding_mask"],
            memory_key_padding_mask=masks["memory_key_padding_mask"]
        )
        
        return decoder_output
    
    @torch.no_grad()
    def inference(self, phoneme, max_decoder_steps=1000, **kwargs):
        device = phoneme.device
        self.eval()  # 추론 모드로 전환하여 dropout 등 비활성화

        if phoneme.dim() == 1:
            phoneme = phoneme.unsqueeze(0)  # [1, max_src_len]

        # 인코더 처리
        encoder_prenet_out = self.encoder_prenet(phoneme)  # [1, src_len, d_model]
        encoder_pe_out = self.scaled_positional_encoding(encoder_prenet_out)
        memory = self.encoder(
            encoder_pe_out,
            mask=None,
            src_key_padding_mask=None,
        )

        # 디코더 초기화
        n_mels = self.decoder.n_mels
        decoded_mel = []
        stop_tokens = []
        stop_threshold = 0.5  # stop token 임계값 (필요에 따라 조정)
        
        # 시작 토큰 설정 (예: 0으로 초기화)
        decoder_input = torch.zeros(1, 1, n_mels).to(device)  # [1, 1, n_mels]

        for t in range(max_decoder_steps):
            # 디코더 프리넷
            decoder_prenet_out = self.decoder_prenet(decoder_input)
            
            # 위치 인코딩
            decoder_pe_out = self.scaled_positional_encoding(decoder_prenet_out)
            
            # 마스크 생성
            tgt_mask = create_square_subsequent_mask(decoder_pe_out.size(1)).to(device)
            
            # 디코더 처리
            decoder_output = self.decoder.decoder(
                tgt=decoder_pe_out,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
            )
            
            # 마지막 타임스텝의 출력 추출
            last_output = decoder_output[:, -1, :]  # [1, d_model]
            
            # 출력 레이어 통과
            mel_output = self.decoder.mel_linear(last_output)  # [1, n_mels]
            stop_token = self.decoder.stop_linear(last_output)  # [1, 1]
            
            # 출력 저장
            decoded_mel.append(mel_output)
            stop_tokens.append(stop_token)

            # stop token 확인
            if F.sigmoid(stop_token).item() > stop_threshold:
                break

            # 다음 디코더 입력 준비
            decoder_input = torch.cat([decoder_input, mel_output.unsqueeze(1)], dim=1)  # [1, t+2, n_mels]

        # 출력들을 하나의 텐서로 결합
        decoded_mel = torch.cat(decoded_mel, dim=0).unsqueeze(0)  # [1, total_steps, n_mels]
        stop_tokens = torch.cat(stop_tokens, dim=0).unsqueeze(0)

        # print(decoded_mel.shape, stop_tokens.shape)

        # PostNet 적용
        mel_output_postnet = self.decoder.postnet(decoded_mel) + decoded_mel  # [1, total_steps, n_mels]

        return decoded_mel, stop_tokens.squeeze(2), mel_output_postnet