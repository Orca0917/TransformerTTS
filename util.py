import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from loguru import logger
from typing import Dict, Tuple, List
from datetime import datetime, timedelta, timezone

# Set Korean Standard Time (KST)
KST = timezone(timedelta(hours=9))


def increment_path(base_path: str) -> str:
    """Create an incremented experiment path."""
    ts = datetime.now(KST).strftime('%m%d-%H%M')
    exp_num = 1

    while True:
        prefix = f"exp_{exp_num}"
        if not any(name.startswith(prefix) for name in os.listdir(base_path)):
            path = os.path.join(base_path, f'{prefix}_{ts}')
            os.makedirs(os.path.join(path, 'mels'), exist_ok=True)
            os.makedirs(os.path.join(path, 'alignments'), exist_ok=True)
            return path
        exp_num += 1


def plot_melspecs(preds: torch.Tensor, targets: torch.Tensor, epoch: int, save_dir: str, valid: bool = True):
    """Plot and save a batch of predicted vs ground-truth mel-spectrograms."""
    preds = preds.detach().cpu().numpy()[:8]
    targets = targets.detach().cpu().numpy()[:8]
    max_len = max(preds.shape[1], targets.shape[1])

    preds = np.pad(preds, ((0, 0), (0, max_len - preds.shape[1]), (0, 0)), constant_values=0)
    targets = np.pad(targets, ((0, 0), (0, max_len - targets.shape[1]), (0, 0)), constant_values=0)
    specs = np.concatenate([preds, targets], axis=0)

    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    for idx, (ax, mel) in enumerate(zip(axes.flatten(), specs)):
        ax.imshow(mel.T, aspect='auto', origin='lower')
        label = f"{'Pred' if idx < 8 else 'GT'} {idx % 8 + 1}"
        color = 'blue' if idx < 8 else 'red'
        ax.text(0.95, 0.95, label, transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold', color=color)
        ax.axis('off')

    plt.tight_layout()
    kind = "valid" if valid else "infer"
    filename = f"{kind}_epoch_{epoch}.png"
    plt.savefig(os.path.join(save_dir, 'mels', filename), dpi=300)
    plt.close(fig)


def plot_melspec(pred: torch.Tensor, target: torch.Tensor, epoch: int, save_dir: str):
    """Plot single predicted and ground-truth mel-spectrograms."""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for i, (mel, title) in enumerate(zip([pred, target], ['Predicted Mel', 'Ground Truth Mel'])):
        axes[i].imshow(mel.T, aspect='auto', origin='lower')
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mels', f"infer_epoch_{epoch}.png"), dpi=300)
    plt.close(fig)


def plot_alignment_grid(alignments: List[torch.Tensor], epoch: int, save_dir: str):
    """Plot attention alignments dynamically based on number of layers and heads."""
    alignments = [a.detach().cpu().numpy() for a in alignments]
    num_layers = len(alignments)
    num_heads = alignments[0].shape[1]

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4 * num_heads, 3 * num_layers))
    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer][head] if num_layers > 1 else axes[head]
            ax.imshow(alignments[layer][0][head].T, aspect='auto', origin='lower')
            ax.set_title(f'Layer {layer + 1} Head {head + 1}')
            ax.set_xlabel('Target Frames')
            ax.set_ylabel('Source Frames')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'alignments', f"valid_epoch_{epoch}.png"), dpi=300)
    plt.close(fig)



def save_specs(pred_mel: torch.Tensor, true_mel: torch.Tensor, lengths: torch.Tensor, epoch: int, save_dir: str, valid: bool = True):
    """Mask padded regions and plot mel-spectrograms."""
    B, T = true_mel.size(0), lengths.max().item()
    mask = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
    pred_mel.masked_fill_(~mask.unsqueeze(-1), 0)
    plot_melspecs(pred_mel, true_mel, epoch, save_dir, valid)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_noam_scheduler(d_model: int, warmup_steps: int):
    """Return Noam learning rate scheduler."""
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr_lambda


def get_teacher_forcing_ratio(epoch: int, total_epochs: int = 300, mode: str = "cosine", **kwargs) -> float:
    """Compute teacher forcing ratio. Supports 'cosine', 'linear', or 'constant'."""
    if mode == "cosine":
        cycles = kwargs.get("cycles", 1)
        A, B = 0.5, 0.5
        return max(min(A * math.cos(math.pi * epoch * cycles / total_epochs) + B, 1.0), 0.95)
    elif mode == "linear":
        return max(1.0 - epoch / total_epochs, 0.05)
    elif mode == "constant":
        return kwargs.get("value", 0.5)
    else:
        raise ValueError(f"Unsupported teacher forcing mode: {mode}")


def prepare_batch(batch: Dict[str, torch.Tensor], device: str) -> Tuple[torch.Tensor, ...]:
    """Move batch to target device."""
    return tuple(batch[k].to(device) for k in ['phoneme', 'melspec', 'phoneme_lens', 'melspec_lens'])


def block_mask(mel: torch.Tensor, p_tf: float, L_bar: int) -> torch.Tensor:
    """Generate block-wise mask for scheduled sampling."""
    B, T, _ = mel.shape
    seed = (torch.rand(B, 1, T, device=mel.device) < (1 - p_tf)).float()
    dilated = F.max_pool1d(seed, kernel_size=L_bar, stride=1, padding=L_bar // 2)
    return dilated.squeeze(1).bool().unsqueeze(-1)[:, :T, :]


def apply_teacher_forcing(pred_melspec: torch.Tensor, melspec: torch.Tensor, melspec_lens: torch.Tensor, p_tf: float, device: str) -> torch.Tensor:
    """Apply block-wise scheduled sampling."""
    mask = block_mask(pred_melspec, p_tf, L_bar=8)
    mel_mixed = torch.where(mask, pred_melspec.detach(), melspec)
    valid = torch.arange(pred_melspec.size(1), device=device).unsqueeze(0) < melspec_lens.unsqueeze(1)
    return mel_mixed * valid.unsqueeze(-1)


def setup_logger(log_path: str = None):
    """Set up loguru logger for console and optional file."""
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.add(log_path, level="DEBUG", rotation="10 MB")
