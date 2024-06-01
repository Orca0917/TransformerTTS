import torch
from dataset import TextMelLoader, TextMelCollate
from config import Config
from torch.utils.data import DataLoader
from model import TransformerTTS
from loss import TransformerTTSLoss
from tqdm import tqdm
from utils.visualization import visualize_spectrograms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(config):
    transformer_tts_dataset = TextMelLoader("./data/metadata.csv", config)
    collate_fn = TextMelCollate()
    dataloader = DataLoader(transformer_tts_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    config = Config()
    model = TransformerTTS(config).to(device)
    criterion = TransformerTTSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # print(len(dataloader))
    epoch = 100
    # 데이터로더에서 데이터를 가져와서 확인하기
    for epoch in range(epoch):
        tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in tqdm_bar:

            phoneme, phoneme_len, mel, stop, mel_len = data
            phoneme = phoneme.to(device)
            mel = mel.to(device)
            stop = stop.to(device)
            phoneme_len = phoneme_len.to(device)
            mel_len = mel_len.to(device)

            post_mel_pred, mel_pred, stop_pred = model(phoneme, phoneme_len, mel, mel_len)

            loss = criterion(post_mel_pred, mel_pred, stop_pred, mel, stop)
            tqdm_bar.set_postfix(epoch=epoch+1, loss=loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                pred = post_mel_pred[0].detach().cpu().numpy().transpose(1, 0)
                targ = mel[0].detach().cpu().numpy().transpose(1, 0)
                # visualize_spectrograms(pred, targ)


if __name__ == "__main__":
    config = Config()
    train(config)