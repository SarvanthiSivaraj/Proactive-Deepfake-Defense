import librosa
import numpy as np


def compute_stft(

        audio,

        n_fft=2048,

        hop_length=512

):

    stft = librosa.stft(

        audio,

        n_fft=n_fft,

        hop_length=hop_length

    )

    return stft


def split_mag_phase(

        stft_matrix
):

    magnitude = np.abs(

        stft_matrix
    )

    phase = np.angle(

        stft_matrix
    )

    return magnitude, phase


def merge_mag_phase(

        magnitude,

        phase
):

    complex_stft = (

        magnitude *

        np.exp(

            1j * phase
        )
    )

    return complex_stft


def inverse_stft(

        stft_matrix,

        hop_length=512
):

    audio = librosa.istft(

        stft_matrix,

        hop_length=hop_length

    )

    return audio