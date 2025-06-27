import os
import librosa
import numpy as np


def load_audio(audio_id, config):
    data_base_path = config['path']['data']
    audio_path = os.path.join(data_base_path, 'wavs', f'{audio_id}.wav')

    sample_rate = config['audio']['sample_rate']
    y, sr = librosa.load(audio_path, sr=sample_rate)
    return y


def get_mel_basis(config):
    mel_basis = librosa.filters.mel(
        sr=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        n_mels=config['audio']['n_mels'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax'],
        norm='slaney',
    )
    return mel_basis


def dynamic_range_compression(S, C=1, clip_val=1e-5):
    S = np.log(np.clip(S, a_min=clip_val, a_max=None) * C)
    return S


def mel_spectrogram(audio_id, config):
    y = load_audio(audio_id, config)
    mel_basis = get_mel_basis(config)

    # STFT
    D = librosa.stft(
        y=y,
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        win_length=config['audio']['win_length'],
        pad_mode='reflect',
        window='hann',
    )
    S = np.abs(D)

    # mel spectrogram
    mel = np.dot(mel_basis, S)
    mel = dynamic_range_compression(mel)

    return mel


def phonemize(transcript, g2p, symbols):

    phonemes = g2p(transcript)
    sequence = phoneme2seq(phonemes, symbols)

    return phonemes, sequence


def phoneme2seq(phoneme, symbols):
    return [symbols.index(p) for p in phoneme if p in symbols]


def seq2phoneme(seq, symbols):
    return [symbols[s] for s in seq if s < len(symbols)]


def normalize(mel, mean, std):
    return (mel - mean[:, None]) / (std[:, None] + 1e-5)


def denormalize(mel, mean, std):
    return (mel * (std[:, None] + 1e-5)) + mean[:, None]