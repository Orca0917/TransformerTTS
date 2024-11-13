import os
import yaml
import torch
import scipy
import argparse
from dataset import TransformerTTSDataset
from torch.utils.data import DataLoader, random_split
from util import seed_everything, get_vocoder, to_device, plot_melspectrogram
from model import TransformerTTS
from loss import TransformerTTSLoss
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def main(m_config, p_config, t_config):
    seed_everything(t_config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # configuration
    batch_size = t_config["training"]["batch_size"]
    lr = t_config["training"]["learning_rate"]
    betas = t_config["training"]["betas"]
    total_step = t_config["training"]["total_step"]

    # dataset
    dataset = TransformerTTSDataset(p_config)
    trainset, validset = random_split(dataset, [len(dataset) - 1310, 1310])
    train_loader = DataLoader(trainset, batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)
    valid_loader = DataLoader(validset, batch_size, shuffle=False, collate_fn=dataset.collate_fn, drop_last=True)

    # trainers
    model = TransformerTTS(m_config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    criterion = TransformerTTSLoss()
    vocoder = get_vocoder(t_config, device)

    global_step_tqdm = tqdm(desc="Global training step", total=total_step)
    global_step_tqdm.update(0)
    epoch = 1

    # train
    while True:
        train_one_epoch(t_config, device, epoch, global_step_tqdm, train_loader, model, criterion, optim, vocoder)
        valid_one_epoch(t_config, device, epoch, global_step_tqdm, valid_loader, model, criterion, vocoder)
        epoch += 1


def train_one_epoch(t_config, device, epoch, global_tqdm, loader, model, criterion, optim, vocoder):
    log_step = t_config["logging"]["log_step"]
    log_path = t_config["general"]["log_path"]
    res_path = t_config["general"]["result_path"]
    synth_step = t_config["logging"]["synth_step"]
    model.train()
    
    global_total_loss = 0
    global_mel_loss = 0
    global_stop_loss = 0
    for batch in tqdm(loader, desc="Training epoch {}".format(epoch)):
        batch = to_device(batch, device)

        # forward
        out = model(**batch)
        loss = criterion(*out, **batch)

        # back propagation
        optim.zero_grad()
        loss["total_loss"].backward()
        optim.step()

        # logging
        global_tqdm.update(1)
        if global_tqdm.n % log_step == 0:
            message1 = "Step {} / {} ".format(global_tqdm.n, global_tqdm.total)
            message2 = "Total loss: {:.4f}, Mel loss: {:.4f}, Stop loss: {:.4f}".format(*loss.values())
            global_tqdm.write(message1 + message2)

            with open(os.path.join(log_path, "log.txt"), "a") as f:
                f.write(message1 + message2 + "\n")

        if global_tqdm.n % synth_step == 0:
            # plot
            tgt_mel = batch["melspectrogram"][0].detach().cpu().numpy()
            prd_mel = out[0][0].detach().cpu().numpy()
            plot_melspectrogram(res_path, global_tqdm.n, tgt_mel, prd_mel)
            
            # synthesize
            y_out = vocoder.inference(out[0][0])
            reconstruction = y_out.view(-1).detach().cpu().numpy()
            wav_path = os.path.join(res_path, "wav", "{}step-{}.wav".format(global_tqdm.n, batch["text"][0]))
            scipy.io.wavfile.write(wav_path, 22050, reconstruction)

        global_total_loss += loss["total_loss"].item()
        global_mel_loss += loss["mel_loss"].item()
        global_stop_loss += loss["stop_loss"].item()

    global_total_loss = global_total_loss / len(loader)
    global_mel_loss = global_mel_loss / len(loader)
    global_stop_loss = global_stop_loss / len(loader)

    message = "[TRAIN] Epoch {:03d} | Average total loss: {:.4f} | Average mel loss: {:.4f} | Average stop loss: {:.4f}".format(
        epoch, global_total_loss, global_mel_loss, global_stop_loss)
    global_tqdm.write(message)


def valid_one_epoch(t_config, device, epoch, global_tqdm, loader, model, criterion, vocoder):
    log_path = t_config["general"]["log_path"]
    res_path = t_config["general"]["result_path"]

    global_total_loss = 0
    global_mel_loss = 0
    global_stop_loss = 0

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Validation epoch {}".format(epoch))):
            batch = to_device(batch, device)

            # forward
            out = model(**batch)
            loss = criterion(*out, **batch)

            global_total_loss += loss["total_loss"].item()
            global_mel_loss += loss["mel_loss"].item()
            global_stop_loss += loss["stop_loss"].item()

    global_total_loss = global_total_loss / len(loader)
    global_mel_loss = global_mel_loss / len(loader)
    global_stop_loss = global_stop_loss / len(loader)

    message = "[VALID] Epoch {:03d} | Average total loss: {:.4f} | Average mel loss: {:.4f} | Average stop loss: {:.4f}".format(
        epoch, global_total_loss, global_mel_loss, global_stop_loss)
    global_tqdm.write(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="path to model configuration file")
    parser.add_argument("-p", "--preprocess", type=str, help="path to preprocess configuration file")
    parser.add_argument("-t", "--train", type=str, help="path to train configuration file")
    args = parser.parse_args()

    m_config = yaml.load(open(args.model, "r"), Loader=yaml.FullLoader)
    p_config = yaml.load(open(args.preprocess, "r"), Loader=yaml.FullLoader)
    t_config = yaml.load(open(args.train, "r"), Loader=yaml.FullLoader)

    main(m_config, p_config, t_config)