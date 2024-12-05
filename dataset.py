import os
import yaml
import torch
import torchaudio
import librosa
import audio

from torch.utils.data import Dataset, DataLoader
from utils.util import process_metadata, pad_1D, pad_2D

class TransformerTTSDataset(Dataset):
    def __init__(self, config):
        super(TransformerTTSDataset, self).__init__()
        self.basepath = config["path"]["corpus_path"]
        self.metadata = process_metadata(os.path.join(config["path"]["metadata_path"], "metadata.csv"))
        self.text_processor = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH.get_text_processor()
        self.audio_processor = audio.AudioProcessor(config)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        audio_name, text = self.metadata[index]
        phoneme, _ = self.text_processor(text)

        audio_path = os.path.join(self.basepath, audio_name + ".wav")
        wav, _ = librosa.load(audio_path)
        melspectrogram = self.audio_processor.get_melspectrogram(wav)  # [80, seq]

        return {
            "text": text,
            "phoneme": phoneme[0].numpy(),
            "melspectrogram": melspectrogram.T,
            "src_len": len(phoneme[0]),
            "mel_len": len(melspectrogram[0]),
        }
    
    def collate_fn(self, batch):
        texts = [data["text"] for data in batch]
        phonemes = [torch.LongTensor(data["phoneme"]) for data in batch]
        melspectrograms = [torch.FloatTensor(data["melspectrogram"]) for data in batch]
        src_lens = torch.LongTensor([data["src_len"] for data in batch])
        mel_lens = torch.LongTensor([data["mel_len"] for data in batch])

        phonemes = pad_1D(phonemes)
        melspectrograms = pad_2D(melspectrograms)

        return texts, phonemes, melspectrograms, src_lens, mel_lens


if __name__ == '__main__':
    config = yaml.load(open("/TransformerTTS/config/preprocess_config.yml", "r"), Loader=yaml.FullLoader)
    dataset = TransformerTTSDataset(config)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)
    for batch in dataloader:
        text, phoneme, melspectrogram, src_len, mel_len = batch
        print("Text sample 1:", text[0])
        print("Phoneme.shape:", phoneme.shape)
        print("Mel-spectrogram.shape:", melspectrogram.shape)
        print("Phoneme lengths:", src_len)
        print("Mel-spectrogram lengths:", mel_len)
        break
