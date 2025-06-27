import os
import yaml
import torch
from loguru import logger

from dataset import DataModule
from util import increment_path, setup_logger
from pytorch_lightning import Trainer, seed_everything
from lightning.lightning_wrapper import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(config_path: str = 'config.yaml'):

    config = yaml.safe_load(open(config_path))
    
    # logging
    exp_dir = increment_path(config['path']['experiment'])
    setup_logger(os.path.join(exp_dir, 'train.log'))
    logger.info(f"Experiment path: {exp_dir}")

    # seed
    seed_everything(42)

    # dataset
    data_module = DataModule(config)
    model = LightningModule(config, exp_dir)
    
    # early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['max_patience'],
        mode='min',
        verbose=True
    )

    trainer = Trainer(
        logger=False,
        max_epochs=config['training']['num_epochs'],
        gradient_clip_val=config['training']['max_grad_norm'],
        accumulate_grad_batches=config['training']['grad_acc_steps'],
        log_every_n_steps=10,
        val_check_interval=1.0,
        default_root_dir=exp_dir,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=False,
        callbacks=[early_stop_callback],
    )
    trainer.fit(model, datamodule=data_module)
    logger.info("Training complete")


if __name__ == '__main__':
    main()
