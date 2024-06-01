from text import text_to_sequence
from scipy.io.wavfile import read
from audio import melspectrogram
import scipy.signal as sps
import numpy as np
import librosa
import torch
import os

# 데이터셋의 음성파일 이름과 transcript(대본) 쌍을 리스트로 반환
def load_filepaths_and_text(metadata_path: str) -> list:

    audiopaths_and_text = []
    with open(metadata_path, "r") as metadata_f:
        lines = metadata_f.readlines()
        for line in lines:
            audiopath, _, text = line.split('|')
            audiopaths_and_text.append([audiopath, text])

    return audiopaths_and_text


# 음성파일을 읽어서 torch 형태로 반환
def load_wav_to_torch(audio_path: str, target_sampling_rate: int=22050):
    # scipy를 사용해서 읽을 경우, 정규화가 되지 않은 데이터 반환
    sampling_rate, data = read(audio_path)

    # 음성의 sampling rate를 target sampling rate에 맞게 변환
    number_of_samples = round(len(data) * float(target_sampling_rate) / sampling_rate)
    sampling_rate = target_sampling_rate
    data = sps.resample(data, number_of_samples)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class TextMelLoader(torch.utils.data.Dataset):
    """
    # TransformerTTS의 custom dataset
    """
    def __init__(self, audiopaths_and_text, config):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.config = config
        self.data_path = config.data_path
        self.max_wav_value = config.max_wav_value

    # 텍스트로부터 encoding된 phoneme(음소) 시퀀스를 반환
    def get_text(self, text):
        text = torch.IntTensor(text_to_sequence(text, ["english_cleaners"]))
        return text
    
    # 오디오 경로를 전달받아 melspectrogram을 반환
    def get_mel(self, audio_path):
        # FloatTensor형태의 audio 반환
        audio, sampling_rate = load_wav_to_torch(audio_path)

        # int16 범위 내의 값들로 정규화 (self.max_wav_value = 32768)
        audio = audio / self.max_wav_value

        # 음성의 시작과 끝에 35dB보다 낮은 음성은 제거 (노이즈로 간주)
        audio, _ = librosa.effects.trim(audio, top_db=35, frame_length=6000, hop_length=200)
        audio = audio.unsqueeze(0)  # (1, seq_len)
        audio = torch.autograd.Variable(audio, requires_grad=False) 

        # 멜스펙트로그램으로 변환 
        melspec = torch.from_numpy(melspectrogram(audio.squeeze(0), self.config))  # (seq_len)
        melspec = torch.squeeze(melspec, 0)  # (1, seq_len)
        return melspec
    
    # 1개의 (phoneme sequence, melspectrogram) 쌍을 반환
    def get_mel_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text
        text = self.get_text(text)
        mel = self.get_mel(os.path.join(self.data_path, audiopath) + '.wav')
        return (text, mel)

    def __len__(self):
        return len(self.audiopaths_and_text)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])
    
    
class TextMelCollate():
    """
    Batch 내의 sequence 길이가 모두 달라 맞춰주는 용도
    """

    def __init__(self):
        ...

    def __call__(self, batch):
        # batch내 각 음소 시퀀스의 길이와 내림차순 순서를 반환
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(text[0]) for text in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        # 0으로 채워진 2차원 tensor 생성 (배치크기, 최대 음소 시퀀스 길이)
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        # 긴 것먼저 batch 데이터 구성하기
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        num_mels = batch[0][1].size(0)  # 80
        max_target_len = max([x[1].size(1) for x in batch])  # 가장 긴 멜스펙트로그램 길이 찾기

        # 멜스펙트로그램과 종료토큰데이터 생성하기
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return (
            text_padded,                    # (B, max_ph_len)
            input_lengths,                  # (B, 1)
            mel_padded.transpose(1, 2),     # (B, max_mel_len, n_mel)
            gate_padded,                    # (B, max_mel_len)
            output_lengths                  # (B, 1)
        )