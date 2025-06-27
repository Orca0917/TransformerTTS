import torch
import pytorch_lightning as pl

from loguru import logger
from typing import Dict, Any
from model.model import TransformerTTS
from loss import TransformerTTSLoss
from util import (
    save_specs,
    plot_melspec,
    prepare_batch,
    get_noam_scheduler,
    plot_alignment_grid,
    apply_teacher_forcing,
    get_teacher_forcing_ratio,
)


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        exp_dir: str = None,
    ):
        super().__init__()
        self.model = TransformerTTS(**config['model'])
        self.criterion = TransformerTTSLoss(**config['loss'])
        self.config = config
        self.exp_dir = exp_dir

        self.train_losses = []
        self.valid_losses = []
        self.example_batch = None
        self.log_interval = config['training'].get('log_interval', 100)


    def forward(self, phoneme, melspec, phoneme_lens, melspec_lens):
        return self.model(phoneme, melspec, phoneme_lens, melspec_lens)


    def training_step(self, batch, batch_idx):

        # process batch
        (
            phoneme, melspec, phoneme_lens, melspec_lens
        ) = prepare_batch(batch, self.device)

        # teacher forcing: use model to predict mel spectrogram
        with torch.no_grad():
            pred_melspec = self.forward(
                phoneme=phoneme,
                melspec=melspec,
                phoneme_lens=phoneme_lens,
                melspec_lens=melspec_lens
            )['pred_melspec']

        # scheduled sampling
        p_tf = get_teacher_forcing_ratio(
            epoch=self.current_epoch + 1,
            total_epochs=self.config['training']['num_epochs'], 
            cycles=1
        )
        mel_mixed = apply_teacher_forcing(
            pred_melspec=pred_melspec,
            melspec=melspec,
            melspec_lens=melspec_lens,
            p_tf=p_tf,
            device=self.device
        )

        # forward pass
        output = self.forward(phoneme, mel_mixed, phoneme_lens, melspec_lens)
        loss = self.criterion(output, melspec, melspec_lens)

        # logging
        if batch_idx % self.log_interval == 0:
            self.log_step_info(batch_idx, 'TRAIN', loss)
        self.train_losses.append(loss['total'].item())

        return loss['total']
    

    def on_train_epoch_start(self):
        self.total_train_batches = len(self.trainer.train_dataloader)


    def on_train_epoch_end(self):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        logger.opt(colors=True).info(
            "[{:3d}] [{:04}/{:04}] | <green>TRAIN</green> | Average Loss: <red>{:9.5f}</red>".format(
                self.current_epoch + 1, self.total_train_batches, self.total_train_batches, avg_train_loss)
        )
        self.train_losses.clear()


    def validation_step(self, batch, batch_idx):
        if self.example_batch is None:
            self.example_batch = batch

        # process batch
        (
            phoneme, melspec, phoneme_lens, melspec_lens
        ) = prepare_batch(batch, self.device)

        # forward pass
        output = self.forward(
            phoneme=phoneme, 
            melspec=melspec, 
            phoneme_lens=phoneme_lens, 
            melspec_lens=melspec_lens,
        )
        loss = self.criterion(output, melspec, melspec_lens)

        # logging
        if batch_idx == 0:
            save_specs(output['pred_melspec'], melspec, melspec_lens, self.current_epoch + 1, self.exp_dir, valid=True)
            plot_alignment_grid(output['alignments'], self.current_epoch + 1, self.exp_dir)

        if batch_idx % self.log_interval == 0:
            self.log_step_info(batch_idx, 'VALID', loss)
        self.valid_losses.append(loss['total'].item())

        return loss['total']
    

    def on_validation_epoch_start(self):
        self.total_valid_batches = len(self.trainer.val_dataloaders)
    

    def on_validation_epoch_end(self):
        avg_valid_loss = sum(self.valid_losses) / len(self.valid_losses)
        logger.opt(colors=True).info(
            "[{:3d}] [{:04}/{:04}] | <yellow>VALID</yellow> | Average Loss: <red>{:9.5f}</red>".format(
                self.current_epoch + 1, self.total_valid_batches, self.total_valid_batches, avg_valid_loss)
        )
        self.log('val_loss', avg_valid_loss, on_epoch=True)
        self.valid_losses.clear()

        if self.example_batch:
            with torch.no_grad():
                # process batch
                (
                    phoneme, melspec, phoneme_lens, _
                ) = prepare_batch(self.example_batch, self.device)

                # forward pass
                pred_melspec = self.model.inference(phoneme[:1], phoneme_lens[:1])['pred_melspec']
                
                # logging
                plot_melspec(pred_melspec, melspec[:1], self.current_epoch + 1, self.exp_dir)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        lr_lambda = get_noam_scheduler(
            d_model=self.config['model']['d_model'],
            warmup_steps=self.config['training']['warmup_steps']
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


    def log_step_info(self, batch_idx: int, phase: str, loss: Dict[str, torch.Tensor]):
        if phase not in ['TRAIN', 'VALID']:
            raise ValueError("phase must be either 'TRAIN' or 'VALID'")

        if phase == 'TRAIN':
            total_batches = self.total_train_batches
            tag_color = 'green'
        else:
            total_batches = self.total_valid_batches
            tag_color = 'yellow'

        if batch_idx % self.log_interval == 0 or batch_idx == total_batches - 1:
            logger.opt(colors=True).info(
                "[{:3d}] [{:04}/{:04}] | <{}>{:>5}</{}> | total: {:9.5f} | mel: {:9.5f} | post: {:9.5f} | stop: {:9.5f} | lr: {: .5E}".format(
                    self.current_epoch + 1,
                    batch_idx,
                    total_batches,
                    tag_color, phase, tag_color,
                    loss['total'].item(),
                    loss['pred_mel'].item(),
                    loss['post_mel'].item(),
                    loss['stop'].item(),
                    self.lr_schedulers().get_last_lr()[0],
                )
            )