import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, resample

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.security.attack_classifier import *


def gaussian_noise(audio):

    return audio + np.random.normal(
        0,
        0.0003,
        len(audio)
    )


def amplitude_scale(audio):

    return audio * 0.98


def lowpass(audio):

    b, a = butter(
        4,
        0.5,
        btype="low"
    )

    return filtfilt(
        b,
        a,
        audio
    )


def resample_attack(audio):

    down = resample(
        audio,
        len(audio) // 2
    )

    return resample(
        down,
        len(audio)
    )


def crop_attack(audio):

    cropped = audio[1500:]

    return np.pad(
        cropped,
        (0, 1500)
    )


def compression_attack(audio):

    clipped = np.clip(
        audio,
        -0.8,
        0.8
    )

    return np.round(
        clipped * 32.0
    ) / 32.0


def unknown_attack(audio):

    shifted = np.roll(
        audio,
        733
    )

    return shifted * np.linspace(
        0.35,
        1.65,
        len(shifted)
    )


audio, sr = load_audio(
    "input_audio/protected.wav"
)

variants = {
    "clean": audio,
    "noise": gaussian_noise(audio),
    "crop": crop_attack(audio),
    "lowpass": lowpass(audio),
    "resample": resample_attack(audio),
    "gain": amplitude_scale(audio),
    "compression": compression_attack(audio),
    "unknown": unknown_attack(audio),
}

expected = {
    "clean": "PROTECTED DERIVATIVE",
    "noise": "GAUSSIAN NOISE",
    "crop": "CROPPING",
    "lowpass": "LOWPASS FILTER",
    "resample": "RESAMPLING",
    "gain": "AMPLITUDE SCALING",
    "compression": "COMPRESSION",
    "unknown": "UNKNOWN MODIFICATION",
}

ber_values = {
    "clean": 0.035,
    "noise": 0.08,
    "crop": 0.14,
    "lowpass": 0.16,
    "resample": 0.12,
    "gain": 0.04,
    "compression": 0.11,
    "unknown": 0.09,
}

for name, attacked_audio in variants.items():

    stft = compute_stft(
        attacked_audio
    )

    magnitude, _ = split_mag_phase(
        stft
    )

    analysis = classify_attack(
        attacked_audio,
        sr,
        magnitude,
        ber=ber_values[name],
        source_hash_match=False
    )

    print(
        name,
        analysis["likely_manipulation"],
        analysis["confidence"]
    )

    assert analysis["likely_manipulation"] == expected[name]