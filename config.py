class Config:
    # default
    base_path = './data'
    data_path = "./data/wavs/"
    preprocess_base_path = base_path + '/preprocessed'

    # preprocess
    max_wav_value = 32768
    sampling_rate=22050
    filter_length=1024
    hop_length=256
    win_length=1024
    n_mel_channels=80
    mel_fmin=0.0
    mel_fmax=8000.0

    preemphasize=True
    preemphasis=0.97
    ref_level_db=20
    min_level_db=-100
    signal_normalization=True
    allow_clipping_in_normalization=True
    symmetric_mels=True
    use_lws=False
    frame_shift_ms=None
    max_abs_value=4.

    # metadata
    data_path = base_path + '/wavs'
    metadata_path = base_path + '/metadata.csv'

    # train
    batch_size = 16

    # model
    num_phonemes = 70
    num_mels = 80
    embedding_dim = 512
    d_model = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # github params
    n_symbols = 71
    n_speakers = 1
    d_model = 512