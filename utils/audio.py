import librosa
import numpy as np

def load_audio(path):
    y, _ = librosa.load(path)
    return y

def pre_emphasis(y, alpha=0.97):
    return np.append(y[0], y[1:] - alpha * y[:-1])

def get_spectrogram(path, n_fft, win_length, hop_length):
    y = load_audio(path)
    y = pre_emphasis(y)
    D = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    return np.abs(D)

def get_melspectrogram(spectrogram, n_mels, sr, fmin, fmax, n_fft):
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_filter, spectrogram ** 2)
    log_mel = librosa.power_to_db(mel)
    return _normalize(log_mel)

def _normalize(x):
    mu, std = np.mean(x), np.std(x)
    return (x - mu) / std