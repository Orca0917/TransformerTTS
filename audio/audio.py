import numpy as np
import librosa

class AudioProcessor():
    def __init__(self, p_config):
        self.sr = p_config["audio"]["sampling_rate"]
        self.preemphasis = p_config["audio"]["preemphasis"]
        self.hop_length = p_config["audio"]["hop_length"]
        self.n_fft = p_config["audio"]["filter_length"]
        self.n_mels = p_config["audio"]["n_mel_channels"]
        self.fmin = p_config["audio"]["mel_fmin"]
        self.fmax = p_config["audio"]["mel_fmax"]

        self.mu = np.load("/TransformerTTS/vocoder/mu.npy")
        self.var = np.load("/TransformerTTS/vocoder/var.npy")
        self.mel_filter = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

    def get_melspectrogram(self, y: np.ndarray):

        # preemphasis
        # if self.preemphasis is not None:
        #     y = np.append(y[0], y[1:] - self.preemphasis * y[:-1])

        # stft
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        D = np.abs(S)

        # mel-spectrogram
        mel = np.dot(self.mel_filter, D)

        # normalizing
        log_mel = np.log10(np.clip(mel, a_min=1e-5, a_max=None))
        melspec = (log_mel - self.mu[:, np.newaxis]) / np.sqrt(self.var[:, np.newaxis])

        return melspec