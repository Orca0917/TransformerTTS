import torch
import torch.nn as nn

class TransformerTTSLoss():
    def __init__(self):
        self.mel_loss = nn.MSELoss()
        self.stop_loss = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for stop token prediction
        self.r_gate = 5.0

    def __call__(self, mel_output, stop_token, mel_output_postnet, melspectrogram, mel_len, **args):
        # Detach targets from computation graph
        melspectrogram = melspectrogram.detach()
        mel_len = mel_len.detach()

        # Calculate mel loss with and without postnet
        mel_loss = self.mel_loss(mel_output, melspectrogram) + self.mel_loss(mel_output_postnet, melspectrogram)

        # Create stop token target using mel_len
        stop_target = (torch.arange(mel_output.size(1), device=mel_output.device) >= mel_len[:, None]).float()

        # Reshape and calculate stop token loss
        stop_loss = self.stop_loss(stop_token, stop_target) * self.r_gate

        return {
            "total_loss": mel_loss + stop_loss,
            "mel_loss": mel_loss,
            "stop_loss": stop_loss
        }