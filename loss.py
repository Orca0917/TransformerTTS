import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TransformerTTSLoss(nn.Module):
    def __init__(
        self,
        stop_weight: float = 8.0,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.register_buffer("pos_weight", torch.tensor(stop_weight))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        mel: torch.Tensor,
        lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pred_melspec  = outputs["pred_melspec"]        # (B, T, C)
        post_melspec  = outputs["post_melspec"]        # (B, T, C)
        pred_stop = outputs["pred_stop"]       # (B, T)

        B, T, C = pred_melspec.size()
        device = lengths.device

        # valid mask: positions before length
        seq_range = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = seq_range < lengths.unsqueeze(1)    # (B, T)
        gate = seq_range == (lengths.unsqueeze(1) - 1)

        # melspectrogram loss
        true_melspec = mel[mask].view(-1, C)
        pred_melspec = pred_melspec[mask].view(-1, C)
        post_melspec = post_melspec[mask].view(-1, C)

        pred_mel_loss = self.mse(pred_melspec, true_melspec)
        post_mel_loss = self.mse(post_melspec, true_melspec)
        mel_loss = pred_mel_loss + 0.5 * post_mel_loss

        # stop token loss
        stop_loss = F.binary_cross_entropy_with_logits(
            pred_stop[mask], gate.float()[mask],
            reduction='mean',
            pos_weight=self.pos_weight,
        )

        total_loss = mel_loss + stop_loss
        return {
            "total": total_loss,
            "pred_mel": pred_mel_loss,
            "post_mel": post_mel_loss,
            "stop": stop_loss,
        }
