import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# 1) 배치 내 상위 8개 Pred vs GT melspec
# ──────────────────────────────────────────────────────────────────────────────
def plot_mels_batch(preds: Tensor, targets: Tensor, epoch: int, save_dir: str):
    preds_np = preds.detach().cpu().numpy()[:8]
    targs_np = targets.detach().cpu().numpy()[:8]
    max_len = max(preds_np.shape[1], targs_np.shape[1])
    preds_pad = np.pad(preds_np, ((0,0),(0,max_len-preds_np.shape[1]),(0,0)), constant_values=0)
    targs_pad = np.pad(targs_np, ((0,0),(0,max_len-targs_np.shape[1]),(0,0)), constant_values=0)
    specs = np.concatenate([preds_pad, targs_pad], axis=0)

    os.makedirs(os.path.join(save_dir, 'mels_batch'), exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16,10))
    for idx, ax in enumerate(axes.flatten()):
        mel = specs[idx]
        ax.imshow(mel.T, aspect='auto', origin='lower')
        label = ('Pred' if idx<8 else 'GT') + f" {idx%8+1}"
        color = 'blue' if idx<8 else 'red'
        ax.text(0.95, 0.95, label, transform=ax.transAxes,
                ha='right', va='top', fontsize=12,
                fontweight='bold', color=color)
        ax.axis('off')
    plt.tight_layout()
    fname = f"valid_epoch_{epoch}.png"
    fig.savefig(os.path.join(save_dir, 'mels_batch', fname), dpi=300)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2) 배치 내 1개 Pred vs GT melspec
# ──────────────────────────────────────────────────────────────────────────────
def plot_mels_single(pred: Tensor, target: Tensor, epoch: int, save_dir: str):
    pred_np = pred.detach().cpu().numpy()
    targ_np = target.detach().cpu().numpy()

    os.makedirs(os.path.join(save_dir, 'mels_single'), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10,6))
    for ax, mel, title in zip(axes, [pred_np, targ_np], ['Predicted', 'Ground Truth']):
        ax.imshow(mel.T, aspect='auto', origin='lower')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    fname = f"infer_epoch_{epoch}.png"
    fig.savefig(os.path.join(save_dir, 'mels_single', fname), dpi=300)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Scheduled sampling 입력 vs 실제 target 비교
# ──────────────────────────────────────────────────────────────────────────────
def plot_mels_scheduled(input_mels: Tensor, targets: Tensor, epoch: int, save_dir: str):
    """
    input_mels: model 입력으로 들어간 mel (예: self-feeding)
    targets: 실제 ground-truth mel
    """
    inp_np = input_mels.detach().cpu().numpy()[:8]
    targ_np = targets.detach().cpu().numpy()[:8]
    max_len = max(inp_np.shape[1], targ_np.shape[1])
    inp_pad = np.pad(inp_np, ((0,0),(0,max_len-inp_np.shape[1]),(0,0)), constant_values=0)
    targ_pad = np.pad(targ_np, ((0,0),(0,max_len-targ_np.shape[1]),(0,0)), constant_values=0)
    specs = np.concatenate([inp_pad, targ_pad], axis=0)

    os.makedirs(os.path.join(save_dir, 'mels_scheduled'), exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16,10))
    for idx, ax in enumerate(axes.flatten()):
        mel = specs[idx]
        ax.imshow(mel.T, aspect='auto', origin='lower')
        label = ('Input' if idx<8 else 'Target') + f" {idx%8+1}"
        color = 'green' if idx<8 else 'orange'
        ax.text(0.95, 0.95, label, transform=ax.transAxes,
                ha='right', va='top', fontsize=12,
                fontweight='bold', color=color)
        ax.axis('off')
    plt.tight_layout()
    fname = f"scheduled_epoch_{epoch}.png"
    fig.savefig(os.path.join(save_dir, 'mels_scheduled', fname), dpi=300)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4) 배치 내 상위 8개 alignment
# ──────────────────────────────────────────────────────────────────────────────
def plot_alignments_batch(alignments: List[Tensor], epoch: int, save_dir: str, top_k: int = 4):
    """
    각 레이어별로 배치 상위 top_k 샘플의 헤드 평균 alignment를 그립니다.
    alignments: List of torch.Tensor, each of shape (B, H, T_out, T_in)
    """
    # NumPy 변환 및 상위 top_k 샘플 선택
    np_alns = [a.detach().cpu().numpy()[:top_k] for a in alignments]
    num_layers = len(np_alns)
    
    # 디렉토리 생성
    save_path = os.path.join(save_dir, 'align_batch')
    os.makedirs(save_path, exist_ok=True)
    
    # 동적 서브플롯: 행=num_layers, 열=top_k
    fig, axes = plt.subplots(
        nrows=num_layers,
        ncols=top_k,
        figsize=(4 * top_k, 3 * num_layers),
        squeeze=False
    )
    
    for lyr, layer_aln in enumerate(np_alns):
        # 헤드 평균: axis=1
        # layer_aln shape = (top_k, H, T_out, T_in)
        avg_aln = layer_aln.mean(axis=1)  # shape = (top_k, T_out, T_in) :contentReference[oaicite:3]{index=3}
        
        for idx in range(top_k):
            ax = axes[lyr][idx]
            ax.imshow(avg_aln[idx].T, aspect='auto', origin='lower')
            ax.set_title(f'Layer {lyr+1} ─ Sample {idx+1}')
            ax.axis('off')
    
    plt.tight_layout()
    fname = f"valid_align_batch_epoch_{epoch}.png"
    fig.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close(fig)



# ──────────────────────────────────────────────────────────────────────────────
# 5) 단일 데이터 alignment (레이어×헤드)
# ──────────────────────────────────────────────────────────────────────────────
def plot_alignment_single(alignments: List[Tensor], idx: int, epoch: int, save_dir: str):
    """
    alignments: list of (B, H, T_out, T_in)
    idx: 시각화할 배치 내 인덱스
    """
    alns = [a.detach().cpu().numpy()[idx] for a in alignments]
    num_layers = len(alns)
    num_heads = alns[0].shape[0]

    os.makedirs(os.path.join(save_dir, 'align_single'), exist_ok=True)
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4*num_heads, 3*num_layers))
    for lyr in range(num_layers):
        for hd in range(num_heads):
            ax = axes[lyr][hd] if num_layers>1 else axes[hd]
            ax.imshow(alns[lyr][hd].T, aspect='auto', origin='lower')
            ax.set_title(f'Layer {lyr+1} Head {hd+1}')
            ax.axis('off')
    plt.tight_layout()
    fname = f"valid_align_{idx}_epoch_{epoch}.png"
    fig.savefig(os.path.join(save_dir, 'align_single', fname), dpi=300)
    plt.close(fig)
