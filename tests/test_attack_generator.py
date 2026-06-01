import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, resample

sys.path.append(os.path.abspath("."))

from src.preprocessing.loader import load_audio


def gaussian_noise(audio):
    return audio + np.random.normal(0, 0.0003, len(audio))


def lowpass(audio):
    b, a = butter(4, 0.5, btype="low")
    return filtfilt(b, a, audio)


def gain(audio):
    return audio * 0.98


def crop(audio):
    cropped = audio[1500:]
    return np.pad(cropped, (0, 1500))


def resample_attack(audio):
    down = resample(audio, len(audio) // 2)
    return resample(down, len(audio))


def compression(audio):
    clipped = np.clip(audio, -0.8, 0.8)
    return np.round(clipped * 32.0) / 32.0


def unknown(audio):
    shifted = np.roll(audio, 733)
    return shifted * np.linspace(0.35, 1.65, len(shifted))


audio, sr = load_audio("input_audio/protected.wav")

variants = {
    "clean": audio,
    "noise": gaussian_noise(audio),
    "lowpass": lowpass(audio),
    "gain": gain(audio),
    "crop": crop(audio),
    "resample": resample_attack(audio),
    "compression": compression(audio),
    "unknown": unknown(audio),
}

for name, variant in variants.items():
    assert len(variant) == len(audio)
    print(name, variant.shape)
