import os
import random
import torch
import scipy
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

def synthesize(vocoder, melspectrogram, save_path, sr=22050):
    y_out = vocoder.inference(melspectrogram)
    reconstruction = y_out.view(-1).detach().cpu().numpy()
    scipy.io.wavfile.write(save_path, 22050, reconstruction)