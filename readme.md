## TransformerTTS

### Overview

This repository is a PyTorch implementation of a neural network-based speech synthesis model, Transformer-TTS, which uses the Transformer Network. It is based on the code by [ChoiHkk's Transformer-TTS](https://github.com/choiHkk/Transformer-TTS/tree/main) and has been trained on the LJSpeech dataset. You can run the [notebook](https://github.com/Orca0917/TransformerTTS/blob/main/TransformerTTS.ipynb) in a Google Colab environment.

<br>

### Model architecture

![Transformer architecture](./asset/transformer-tts-architecture.png)

<br>

### Dataset

The dataset used is the English speech dataset LJSpeech. In the Jupyter notebook, the dataset is utilized without a separate download by using the `torchaudio` package. The data preprocessing was implemented by referring to the tacotron audio preprocessing by [Kyubong Park](https://github.com/Kyubyong).

* https://keithito.com/LJ-Speech-Dataset/
* torchaudio.dataset

<br>

### Result 

The training was conducted with a batch size of 16 on a total of 13,100 voice datasets for 10 epochs. The result is expressed as a gif showing the predicted mel spectrogram and the ground truth mel spectrogram every 100 steps.

![training result](./asset/transformer-tts-result.gif)

<br>

### Dependency

```text
torch                            2.3.0+cu121
torchaudio                       2.3.0+cu121
librosa                          0.10.2.post1
numpy                            1.25.2
scipy                            1.11.4
python                           3.10.12
```

<br>

### References

* Li, N., Liu, S., Liu, Y., Zhao, S., & Liu, M. (2019, July). Neural speech synthesis with transformer network. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 6706-6713).

* keithito https://github.com/keithito/tacotron/tree/master/text

* ming024 https://github.com/ming024/FastSpeech2/tree/master/audio

* choiHkk https://github.com/choiHkk/Transformer-TTS/tree/main

