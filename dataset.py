import os
import yaml
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils.util import process_metadata, plot_mel, pad_1D, pad_2D
from utils.audio import get_spectrogram, get_melspectrogram

class TransformerTTSDataset(Dataset):
    def __init__(self, config):
        super(TransformerTTSDataset, self).__init__()
        self.basepath = config["path"]["corpus_path"]
        self.metadata = process_metadata(os.path.join(config["path"]["metadata_path"], "metadata.csv"))
        self.text_processor = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH.get_text_processor()

        self.sr = config["preprocessing"]["audio"]["sampling_rate"]
        self.n_fft = config["preprocessing"]["stft"]["n_fft"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]

        self.n_mels = config["preprocessing"]["mel"]["n_mels"]
        self.fmin = config["preprocessing"]["mel"]["fmin"]
        self.fmax = config["preprocessing"]["mel"]["fmax"]

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        audio_name, text = self.metadata[index]
        phoneme, _ = self.text_processor(text)

        audio_path = os.path.join(self.basepath, audio_name + ".wav")
        spectrogram = get_spectrogram(audio_path, self.n_fft, self.win_length, self.hop_length)
        melspectrogram = get_melspectrogram(spectrogram, self.n_mels, self.sr, self.fmin, self.fmax, self.n_fft)

        return {
            "text": text,
            "phoneme": phoneme[0].numpy(),
            "spectrogram": spectrogram.T,
            "melspectrogram": melspectrogram.T,
            "src_len": len(phoneme[0]),
            "mel_len": len(melspectrogram[0]),
        }
    
    def collate_fn(self, batch):
        texts = [data["text"] for data in batch]
        phonemes = [torch.LongTensor(data["phoneme"]) for data in batch]
        spectrograms = [torch.FloatTensor(data["spectrogram"]) for data in batch]
        melspectrograms = [torch.FloatTensor(data["melspectrogram"]) for data in batch]
        src_lens = torch.LongTensor([data["src_len"] for data in batch])
        mel_lens = torch.LongTensor([data["mel_len"] for data in batch])

        phonemes = pad_1D(phonemes)
        spectrograms = pad_2D(spectrograms)
        melspectrograms = pad_2D(melspectrograms)

        return texts, phonemes, spectrograms, melspectrograms, src_lens, mel_lens


if __name__ == '__main__':
    config = yaml.load(open("/TransformerTTS/config/preprocess_config.yml", "r"), Loader=yaml.FullLoader)
    dataset = TransformerTTSDataset(config)

    plot_mel(dataset[0]["melspectrogram"])

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)
    for batch in dataloader:
        text, phoneme, spectrogram, melspectrogram, src_len, mel_len = batch
        print("Text sample 1:", text[0])
        print("Phoneme.shape:", phoneme.shape)
        print("Spectrogram.shape:", spectrogram.shape)
        print("Mel-spectrogram.shape:", melspectrogram.shape)
        print("Phoneme lengths:", src_len)
        print("Mel-spectrogram lengths:", mel_len)
        break
