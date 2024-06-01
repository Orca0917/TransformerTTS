import torch.nn as nn

class TransformerTTSLoss():
    def __init__(self):
        self.mel_loss = nn.MSELoss()
        self.stop_loss = nn.BCEWithLogitsLoss()
        self.r_gate = 5.0

    def __call__(self, post_mel_pred, mel_pred, stop_pred, mel_target, stop_target):
        mel_target.requires_grad = False
        stop_target.requires_grad = False
        stop_target = stop_target.view(-1, 1)

        stop_pred = stop_pred.view(-1, 1)
        mel_loss = self.mel_loss(mel_pred, mel_target) + \
            self.mel_loss(post_mel_pred, mel_target)
        gate_loss = self.stop_loss(stop_pred, stop_target) * self.r_gate
        return mel_loss + gate_loss