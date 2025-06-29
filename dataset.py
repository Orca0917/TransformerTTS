import os
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Literal

import torch
from torch.utils.data import Dataset

class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['training'].get('batch_size', 1)
        self.num_workers = config['training'].get('num_workers', 2)

    def setup(self, stage=None):
        self.train_dataset = TransformerTTSDataset(self.config, mode='train')
        self.valid_dataset = TransformerTTSDataset(self.config, mode='valid')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )


class TransformerTTSDataset(Dataset):
    def __init__(self, config: Dict, mode: Literal['train', 'valid'] = 'train'):
        super(TransformerTTSDataset, self).__init__()
        self.data_dir = config['path']['preprocessed']
        self.mode = mode

        # Load file list
        all_files = sorted(f for f in os.listdir(self.data_dir) if f.endswith('.npz'))
        valid_prefixes = ('LJ001', 'LJ002', 'LJ003')

        self.data_list = [
            f for f in all_files
            if (f.startswith(valid_prefixes) if mode == 'valid' else not f.startswith(valid_prefixes))
        ]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_list[idx])
        data = np.load(file_path, allow_pickle=True)
        return {
            'transcript': str(data['transcript']),
            'melspec': torch.from_numpy(data['melspec']).T,
            'phoneme': torch.from_numpy(data['sequence']),
        }


def collate_fn(batch):

    # Sort by sequence length (descending)
    phoneme_lens = torch.tensor([len(sample['phoneme']) for sample in batch])
    sorted_indices = torch.argsort(phoneme_lens, descending=True)
    batch = [batch[i] for i in sorted_indices]

    max_input_len = phoneme_lens[sorted_indices[0]]
    max_target_len = max(sample['melspec'].shape[0] for sample in batch)
    n_mels = batch[0]['melspec'].shape[1]

    # Allocate padded tensors
    phoneme_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    melspec_padded = torch.zeros(len(batch), max_target_len, n_mels)
    melspec_lens = torch.zeros(len(batch), dtype=torch.long)
    transcripts = []

    for i, sample in enumerate(batch):
        seq_len = len(sample['phoneme'])
        mel_len = sample['melspec'].shape[0]

        phoneme_padded[i, :seq_len] = sample['phoneme']
        melspec_padded[i, :mel_len, :] = sample['melspec']
        melspec_lens[i] = mel_len
        transcripts.append(sample['transcript'])

    return {
        'phoneme': phoneme_padded,
        'melspec': melspec_padded,
        'phoneme_lens': phoneme_lens[sorted_indices],
        'melspec_lens': melspec_lens,
        'transcript': transcripts,
    }