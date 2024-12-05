import librosa
import numpy as np
from scipy import signal

class AudioProcessor:
    def __init__(self, p_config):
        self.sampling_rate          = p_config["audio"]["sampling_rate"]
        self.preemphasis            = p_config["audio"]["preemphasis"]
        self.frame_shift_ms         = p_config["audio"]["frame_shift_ms"]
        self.hop_length             = p_config["audio"]["hop_length"]
        self.filter_length          = p_config["audio"]["filter_length"]
        self.win_length             = p_config["audio"]["win_length"]
        self.n_mel_channels         = p_config["audio"]["n_mel_channels"]
        self.mel_fmin               = p_config["audio"]["mel_fmin"]
        self.mel_fmax               = p_config["audio"]["mel_fmax"]
        self.ref_level_db           = p_config["audio"]["ref_level_db"]
        self.min_level_db           = p_config["audio"]["min_level_db"]
        self.max_abs_value          = p_config["audio"]["max_abs_value"]
        self.signal_normalization   = p_config["audio"]["signal_normalization"]
        self.mel_basis              = self._build_mel_basis()

    def apply_preemphasis(self, wav, preemphasize=True):
        k = self.preemphasis
        return signal.lfilter([1, -k], [1], wav) if preemphasize else wav

    def get_hop_size(self):
        if self.hop_length is not None:
            return self.hop_length
        assert self.frame_shift_ms is not None
        return int(self.frame_shift_ms / 1000 * self.sampling_rate)
    
    def melspectrogram(self, wav):
        D = self._stft(self.apply_preemphasis(wav, self.preemphasis))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S) if self.signal_normalization else S
    
    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.filter_length,
            hop_length=self.get_hop_size(),
            win_length=self.win_length,
            pad_mode='constant'
        )
    
    def _linear_to_mel(self, spectrogram):
        return np.dot(self.mel_basis, spectrogram)
    
    def _build_mel_basis(self):
        return librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.filter_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
    
    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))
    
    def _normalize(self, S):
        return np.clip(
            (2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
            -self.max_abs_value,
            self.max_abs_value
        )
    
    def _denormalize(self, D):
        return (((np.clip(D, -self.max_abs_value, self.max_abs_value) + self.max_abs_value)
                 * -self.min_level_db / (2 * self.max_abs_value)) + self.min_level_db)