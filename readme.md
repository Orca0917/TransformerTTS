## TransformerTTS

### 개요

이 레포지토리는 Transformer Network를 사용한 신경망 기반 음성 합성 모델인 Transformer-TTS의 PyTorch 구현입니다. ChoiHkk님의 [Transformer-TTS](https://github.com/choiHkk/Transformer-TTS/tree/main) 코드를 기반으로 구현되었으며, LJSpeech 데이터셋을 사용하여 학습되었습니다. Google colab 환경에서 [노트북](https://github.com/Orca0917/TransformerTTS/blob/main/TransformerTTS.ipynb)파일을 실행시켜볼 수 있습니다.

### 데이터셋

* https://keithito.com/LJ-Speech-Dataset/
* torchaudio.dataset

### 결과 

![training result](./asset/transformer-tts-result.gif)


### 참고 자료

* Li, N., Liu, S., Liu, Y., Zhao, S., & Liu, M. (2019, July). Neural speech synthesis with transformer network. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 6706-6713).

* keithito https://github.com/keithito/tacotron/tree/master/text

* ming024 https://github.com/ming024/FastSpeech2/tree/master/audio

* choiHkk https://github.com/choiHkk/Transformer-TTS/tree/main

