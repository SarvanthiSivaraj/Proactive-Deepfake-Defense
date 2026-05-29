import librosa
import soundfile as sf
import numpy as np


def load_audio(filepath, sr=44100):

    audio, sr = librosa.load(
        filepath,
        sr=sr,
        mono=True
    )

    return audio, sr


def normalize_audio(audio):

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak


def save_audio(filepath, audio, sr):

    sf.write(
        filepath,
        audio,
        sr
    )