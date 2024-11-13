import librosa
import numpy as np

def load_audio(path):
    y, _ = librosa.load(path)
    return y

def get_spectrogram(path, n_fft, win_length, hop_length):
    y = load_audio(path)
    D = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    return np.abs(D)

def get_melspectrogram(spectrogram, n_mels, sr, fmin, fmax, n_fft):
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_filter, spectrogram)
    log_mel = np.log(np.clip(mel, a_min=0.01, a_max=None))
    return _normalize(log_mel)

def _normalize(x):
    mu, std = np.mean(x), np.std(x)
    return (x - mu) / std