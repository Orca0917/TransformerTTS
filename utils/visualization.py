import matplotlib.pyplot as plt


def visualize_spectrograms(mel_pred, mel_out):
    """
    모델로 예측한 melspectrogram과 정답 melspectrogram 을 시각화
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # 예측한 melspectrogram 그리기
    im1 = ax1.imshow(mel_pred, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title("Prediction")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Channels")

    # 정답 melspectrogram 그리기
    im2 = ax2.imshow(mel_out, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title("Target")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Channels")

    plt.tight_layout()
    plt.show()