import os
import yaml
import json
import numpy as np
from tqdm import tqdm
from g2p_en import G2p
from loguru import logger
from audio import mel_spectrogram, phonemize, normalize


def load_transcripts(metadata_path):
    transcripts = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            audio_id, _, transcript = line.strip().split('|')
            transcripts[audio_id] = transcript.strip()
    logger.info(f"Loaded {len(transcripts)} transcripts.")
    return transcripts


def initialize_g2p():
    g2p = G2p()
    punc = '!\'(),.:;? '
    symbols = g2p.phonemes + list(punc)
    return g2p, symbols


def process_and_save(audio_id, transcript, g2p, symbols, config, out_dir, mean, std):
    phoneme, sequence = phonemize(transcript, g2p, symbols)
    melspec = mel_spectrogram(audio_id, config)

    # Normalize melspec spectrogram
    if mean is not None and std is not None:
        melspec = normalize(melspec, mean, std)

    np.savez(
        os.path.join(out_dir, f'{audio_id}.npz'),
        melspec=melspec,
        transcript=transcript,
        phoneme=np.array(phoneme),
        sequence=np.array(sequence),
    )


def compute_global_stats(transcripts, config):
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for audio_id in tqdm(transcripts, desc="Computing global mel stats"):
        mel = mel_spectrogram(audio_id, config)     # mel   :(#mel, #frame)

        n_mels, T = mel.shape
        total_count += n_mels * T

        total_sum += mel.sum()
        total_sq_sum += (mel ** 2).sum()
    
    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = np.sqrt(var + 1e-8)

    logger.info("Computed global mel mean/std.")
    stats = {
        "mean": mean.item() if isinstance(mean, np.generic) else float(mean),
        "std" : std.item()  if isinstance(std,  np.generic) else float(std),
    }
    with open("stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved global stats to stats.json")

    return mean, std


def preprocess(config):

    out_dir = config['path']['preprocessed']
    os.makedirs(out_dir, exist_ok=True)

    if os.listdir(out_dir):
        logger.info("Preprocessed data already exists. Skipping preprocessing.")
        return

    metadata_path = os.path.join(config['path']['data'], 'metadata.csv')
    if not os.path.isfile(metadata_path):
        logger.error(f"Metadata file not found at {metadata_path}")
        return

    transcripts = load_transcripts(metadata_path)
    g2p, symbols = initialize_g2p()

    mean, std = None, None
    if config['audio']['normalize_mel']:
        mean, std = compute_global_stats(transcripts, config)

    for audio_id, transcript in tqdm(transcripts.items(), desc="Preprocessing"):
        try:
            process_and_save(audio_id, transcript, g2p, symbols, config, out_dir, mean, std)
        except Exception as e:
            logger.warning(f"Error processing {audio_id}: {e}")


if __name__ == '__main__':
    config_path = 'config.yaml'
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found at {config_path}")
        exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    preprocess(config)
