import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import Dict, Tuple, List
from datetime import datetime, timedelta, timezone

import torch
import torch.nn.functional as F
from torch import Tensor

# Set Korean Standard Time (KST)
KST = timezone(timedelta(hours=9))


def increment_path(base_path: str) -> str:
    '''
    Create an incremented experiment path.
    '''
    ts = datetime.now(KST).strftime('%m%d-%H%M')
    exp_num = 1

    while True:
        prefix = f"exp_{exp_num}"
        if not any(name.startswith(prefix) for name in os.listdir(base_path)):
            path = os.path.join(base_path, f'{prefix}_{ts}')
            os.makedirs(os.path.join(path, 'mels_batch'), exist_ok=True)
            os.makedirs(os.path.join(path, 'mels_single'), exist_ok=True)
            os.makedirs(os.path.join(path, 'align_batch'), exist_ok=True)
            os.makedirs(os.path.join(path, 'align_single'), exist_ok=True)
            os.makedirs(os.path.join(path, 'mels_scheduled'), exist_ok=True)
            return path
        exp_num += 1


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_noam_scheduler(d_model: int, warmup_steps: int):
    '''
    Return Noam learning rate scheduler.
    '''
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr_lambda


import math

def get_teacher_forcing_ratio(
    epoch: int,
    total_epochs: int = 300,
    mode: str = "cosine",
    warmup_epochs: int = 10,
    **kwargs
) -> float:
    """
    Compute teacher forcing ratio with warmup.
      - epoch < warmup_epochs: ratio == 1.0
      - epoch >= warmup_epochs: cosine or other schedule
    Supports 'cosine', 'linear', or 'constant'.
    """
    # 1) Warmup 단계: 비율 1 고정
    if epoch < warmup_epochs:
        return 1.0

    # 2) 스케줄 적용을 위한 epoch 재계산
    epoch_adj = epoch - warmup_epochs
    total_adj = max(total_epochs - warmup_epochs, 1)

    if mode == "cosine":
        cycles = kwargs.get("cycles", 1)
        A, B = 0.5, 0.5
        # 코사인 단계 적용
        ratio = A * math.cos(math.pi * epoch_adj * cycles / total_adj) + B
        # [B, 1.0] 사이로 클램프
        return max(min(ratio, 1.0), B)

    elif mode == "linear":
        # 선형 감소: [0.05, 1.0] → warmup 이후 선형 감소 시작
        ratio = 1.0 - epoch_adj / total_adj
        return max(ratio, 0.05)

    elif mode == "constant":
        return kwargs.get("value", 1.0)

    else:
        raise ValueError(f"Unsupported teacher forcing mode: {mode}")



def prepare_batch(batch: Dict[str, Tensor], device: str) -> Tuple[Tensor, ...]:
    '''
    Move batch to target device.
    '''
    return tuple(batch[k].to(device) for k in ['phoneme', 'melspec', 'phoneme_lens', 'melspec_lens'])


def block_mask(mel: Tensor, p_tf: float, L_bar: int) -> Tensor:
    '''
    Generate block-wise mask for scheduled sampling.
    '''
    B, T, _ = mel.shape
    seed = (torch.rand(B, 1, T, device=mel.device) < (1 - p_tf)).float()
    dilated = F.max_pool1d(seed, kernel_size=L_bar, stride=1, padding=L_bar // 2)
    return dilated.squeeze(1).bool().unsqueeze(-1)[:, :T, :]


def apply_teacher_forcing(pred_melspec: Tensor, melspec: Tensor, melspec_lens: Tensor, p_tf: float, device: str) -> Tensor:
    '''
    Apply block-wise scheduled sampling.
    '''
    mask = block_mask(pred_melspec, p_tf, L_bar=8)
    mel_mixed = torch.where(mask, pred_melspec.detach(), melspec)
    valid = torch.arange(pred_melspec.size(1), device=device).unsqueeze(0) < melspec_lens.unsqueeze(1)
    return mel_mixed * valid.unsqueeze(-1)


def setup_logger(log_path: str = None):
    '''
    Set up loguru logger for console and optional file.
    '''
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.add(log_path, level="DEBUG", rotation="10 MB")
