import os
import yaml
import torch
import argparse
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter 

from dataset import TransformerTTSDataset
from utils.util import (
    seed_everything,
    get_vocoder,
    to_device,
    plot_melspectrogram,
    synthesize,
)
from model import TransformerTTS
from loss import TransformerTTSLoss
from optimizer import ScheduledOptim

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, m_config, p_config, t_config):
        self.m_config = m_config
        self.p_config = p_config
        self.t_config = t_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Seed and prepare environment
        seed_everything(t_config)
        self._prepare_directories()
        self._load_data()
        self._initialize_model()
        self.global_step = 0

        # Tensorboard configuration
        self.writer = SummaryWriter(log_dir=self.log_path)

    def _prepare_directories(self):
        self.log_path = self.t_config["general"]["log_path"]
        self.result_path = self.t_config["general"]["result_path"]
        self.ckpt_path = self.t_config["general"]["ckpt_path"]

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "wav"), exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)

    def _load_data(self):
        dataset = TransformerTTSDataset(self.p_config)
        val_size = self.t_config["training"].get("validation_size", 1310)
        train_size = len(dataset) - val_size
        trainset, validset = random_split(dataset, [train_size, val_size])

        batch_size = self.t_config["training"]["batch_size"]
        collate_fn = dataset.collate_fn

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.valid_loader = DataLoader(
            validset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def _initialize_model(self):
        self.model = TransformerTTS(self.m_config).to(self.device)
        self.optimizer = ScheduledOptim(self.model, self.t_config, self.m_config)
        self.criterion = TransformerTTSLoss()
        self.vocoder = get_vocoder(self.t_config, self.device)

        self.total_step = self.t_config["training"]["total_step"]
        self.log_step = self.t_config["logging"]["log_step"]
        self.synth_step = self.t_config["logging"]["synth_step"]
        self.save_step = self.t_config["logging"]["save_step"]

    def train(self):
        epoch = 1
        pbar = tqdm(total=self.total_step, desc="Global Training Step")
        while self.global_step < self.total_step:
            self._train_one_epoch(epoch, pbar)
            self._validate_one_epoch(epoch)
            epoch += 1
        pbar.close()
        self.writer.close()

    def _train_one_epoch(self, epoch, pbar):
        self.model.train()
        total_loss, mel_loss, stop_loss = 0, 0, 0

        for batch in tqdm(
            self.train_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            batch = to_device(batch, self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(**batch)
            losses = self.criterion(*outputs, **batch)

            # Backward and optimize
            losses["total_loss"].backward()
            self.optimizer.step_and_update_lr()
            self.current_lr = self.optimizer._optimizer.param_groups[0]["lr"]

            # Logging
            self.global_step += 1
            pbar.update(1)

            if self.global_step % self.log_step == 0:
                self._log_step(losses, self.current_lr)

            self.writer.add_scalar("Loss/Train/Total_Loss", losses["total_loss"].item(), self.global_step)
            self.writer.add_scalar("Loss/Train/Mel_Loss", losses["mel_loss"].item(), self.global_step)
            self.writer.add_scalar("Loss/Train/Stop_Loss", losses["stop_loss"].item(), self.global_step)
            self.writer.add_scalar("Learning_Rate", self.current_lr, self.global_step)

            if self.global_step % self.save_step == 0:
                self._save_checkpoint()

            # Accumulate losses
            total_loss += losses["total_loss"].item()
            mel_loss += losses["mel_loss"].item()
            stop_loss += losses["stop_loss"].item()

            if self.global_step >= self.total_step:
                break

        # Epoch summary
        self._log_epoch(
            epoch, total_loss, mel_loss, stop_loss, len(self.train_loader), "TRAIN"
        )

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss, mel_loss, stop_loss = 0, 0, 0

        with torch.no_grad():
            for idx, batch in enumerate(
                tqdm(self.valid_loader, desc=f"Validation Epoch {epoch}", leave=False)
            ):
                batch = to_device(batch, self.device)

                # Forward pass
                outputs = self.model(**batch)
                losses = self.criterion(*outputs, **batch)

                self.writer.add_scalar("Loss/Validation/Total_Loss", losses["total_loss"].item(), self.global_step)
                self.writer.add_scalar("Loss/Validation/Mel_Loss", losses["mel_loss"].item(), self.global_step)
                self.writer.add_scalar("Loss/Validation/Stop_Loss", losses["stop_loss"].item(), self.global_step)

                # Accumulate losses
                total_loss += losses["total_loss"].item()
                mel_loss += losses["mel_loss"].item()
                stop_loss += losses["stop_loss"].item()

                if idx == 0:
                    self._synthesize_sample(batch, outputs, validation=True)

        # Epoch summary
        self._log_epoch(
            epoch, total_loss, mel_loss, stop_loss, len(self.valid_loader), "VALID"
        )

    def _log_step(self, losses, current_lr):
        message = (
            f"Step {self.global_step}/{self.total_step} | "
            f"Total Loss: {losses['total_loss']:.4f} | "
            f"Mel Loss: {losses['mel_loss']:.4f} | "
            f"Stop Loss: {losses['stop_loss']:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        tqdm.write(message)
        self._write_log(message)

    def _log_epoch(self, epoch, total_loss, mel_loss, stop_loss, num_batches, mode):
        avg_total_loss = total_loss / num_batches
        avg_mel_loss = mel_loss / num_batches
        avg_stop_loss = stop_loss / num_batches

        message = (
            f"[{mode}] Epoch {epoch:03d} | "
            f"Average Total Loss: {avg_total_loss:.4f} | "
            f"Average Mel Loss: {avg_mel_loss:.4f} | "
            f"Average Stop Loss: {avg_stop_loss:.4f}"
        )
        tqdm.write(message)
        self._write_log(message)

        self.writer.add_scalar(f"Loss/{mode}/Average_Total_Loss", avg_total_loss, epoch)
        self.writer.add_scalar(f"Loss/{mode}/Average_Mel_Loss", avg_mel_loss, epoch)
        self.writer.add_scalar(f"Loss/{mode}/Average_Stop_Loss", avg_stop_loss, epoch)

    def _write_log(self, message):
        with open(os.path.join(self.log_path, "log.txt"), "a") as f:
            f.write(message + "\n")

    def _synthesize_sample(self, batch, outputs, validation=False):
        idx = 0  # Take the first sample in the batch
        tgt_mel = batch["melspectrogram"][idx].detach().cpu().numpy()
        prd_mel = outputs[0][idx].detach().cpu().numpy()

        # Plot mel-spectrogram
        plot_melspectrogram(self.result_path, self.global_step, tgt_mel, prd_mel)

        # Synthesize audio
        text = batch["text"][idx]
        wav_dir = os.path.join(self.result_path, "wav")
        wav_filename = f"{self.global_step}step_{text}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)
        synthesize(self.vocoder, outputs[0][idx], wav_path)

    def _save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer._optimizer.state_dict(),
            "step": self.global_step,
            "learning_rate": self.current_lr,
        }
        save_path = os.path.join(self.ckpt_path, f"{self.global_step}steps.ckpt")
        torch.save(checkpoint, save_path)
        tqdm.write(f"Saved checkpoint at {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer TTS Training Script")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model configuration file")
    parser.add_argument("-p", "--preprocess", type=str, required=True, help="Path to preprocess configuration file")
    parser.add_argument("-t", "--train", type=str, required=True, help="Path to train configuration file")
    return parser.parse_args()


def load_configs(args):
    with open(args.model, "r") as m_file, open(args.preprocess, "r") as p_file, open(
        args.train, "r"
    ) as t_file:
        m_config = yaml.safe_load(m_file)
        p_config = yaml.safe_load(p_file)
        t_config = yaml.safe_load(t_file)
    return m_config, p_config, t_config


if __name__ == "__main__":
    args = parse_args()
    m_config, p_config, t_config = load_configs(args)
    trainer = Trainer(m_config, p_config, t_config)
    trainer.train()
