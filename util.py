import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from parallel_wavegan.utils import load_model

def seed_everything(t_config):
    seed = t_config["general"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_vocoder(t_config, device):
    vocoder_path = t_config["general"]["vocoder_path"]
    vocoder = load_model(vocoder_path).to(device).eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()
    return vocoder


def to_device(batch, device):
    (
        text,
        phoneme,
        spectrogram,
        melspectrogram,
        src_len,
        mel_len
    ) = batch

    phoneme = phoneme.to(device)
    melspectrogram = melspectrogram.to(device)
    src_len = src_len.to(device)
    mel_len = mel_len.to(device)

    return {
        "text": text,
        "phoneme": phoneme,
        "spectrogram": spectrogram,
        "melspectrogram": melspectrogram,
        "src_len": src_len,
        "mel_len": mel_len
    }


def process_metadata(path):
    metadata = []
    with open(path, 'r') as f:
        for line in f.readlines():
            audio_id, _, text = line.strip().split("|")
            metadata.append((audio_id, text))
        
    return metadata

def plot_mel(result_path, step, mel):
    plt.imshow(mel.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel("mel frame")
    plt.ylabel("n_mels")
    plt.tick_params(labelsize="x-small")

    img_path = os.path.join(result_path, "melspectrogram", "{}step.png".format(step))
    plt.savefig(img_path)
    plt.close()


def plot_melspectrogram(result_path, step, targ_mel, pred_mel=None, is_train=True):
    if pred_mel is None:
        plot_mel(img_path, step, targ_mel)
    
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        ax1.imshow(targ_mel.T, origin='lower', aspect='auto', interpolation=None)
        ax1.set_xlabel("mel frame")
        ax1.set_ylabel("n mels")

        ax2.imshow(pred_mel.T, origin='lower', aspect='auto', interpolation=None)
        ax2.set_xlabel("mel frame")
        ax2.set_ylabel("n mels")

        if is_train:
            img_path = os.path.join(result_path, "melspectrogram", "{}step.png".format(step))
        else:
            img_path = os.path.join(result_path, "melspectrogram", "{}step_val.png".format(step))
            
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        


def pad_1D(batch):
    return pad_2D(batch)
    

def pad_2D(batch):
    # batch : [batch_size, seq_len, n_mel]
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

def create_masks(phoneme, melspectrogram, src_len, mel_len):
    device = phoneme.device
    batch_size = phoneme.size(0)
    max_src_len = phoneme.size(1)
    max_mel_len = melspectrogram.size(1)

    # 인코더 마스크
    src_key_padding_mask = create_src_mask(phoneme, src_len)  # Shape: [batch_size, max_src_len]
    encoder_attention_mask = None  # 필요하지 않다면 None으로 설정

    # 디코더 마스크
    tgt_mask = create_look_ahead_mask(max_mel_len, device)  # Shape: [max_mel_len, max_mel_len]
    tgt_key_padding_mask = create_tgt_padding_mask(mel_len, max_mel_len, device)  # Shape: [batch_size, max_mel_len]

    # 크로스 어텐션 마스크 (필요한 경우)
    memory_key_padding_mask = src_key_padding_mask  # Shape: [batch_size, max_src_len]
    memory_mask = None  # 필요하지 않다면 None으로 설정

    return {
        "encoder_attention_mask": encoder_attention_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "tgt_mask": tgt_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask,
        "memory_mask": memory_mask,
        "memory_key_padding_mask": memory_key_padding_mask
    }

def create_src_mask(src_seq, src_len):
    batch_size, max_src_len = src_seq.size()
    src_key_padding_mask = torch.zeros((batch_size, max_src_len), dtype=torch.bool, device=src_seq.device)
    for idx, length in enumerate(src_len):
        src_key_padding_mask[idx, length:] = True
    return src_key_padding_mask

def create_look_ahead_mask(max_len, device):
    return torch.triu(torch.ones((max_len, max_len), device=device, dtype=torch.bool), diagonal=1)

def create_tgt_padding_mask(mel_len, max_len, device):
    batch_size = len(mel_len)
    tgt_padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    for idx, length in enumerate(mel_len):
        tgt_padding_mask[idx, length:] = True
    return tgt_padding_mask

def create_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) == 1, diagonal=1).transpose(0, 1).bool()

