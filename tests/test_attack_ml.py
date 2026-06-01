import os
import sys

sys.path.append(os.path.abspath("."))

from src.forensic.ml import classify_attack_ml, train_attack_ml_model
from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase


def noise(audio):
    import numpy as np

    return audio + np.random.normal(0, 0.0003, len(audio))


def gain(audio):
    return audio * 0.98


def crop(audio):
    import numpy as np

    cropped = audio[1500:]
    return np.pad(cropped, (0, 1500))


def lowpass(audio):
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 0.5, btype="low")
    return filtfilt(b, a, audio)


def resample_attack(audio):
    from scipy.signal import resample

    down = resample(audio, len(audio) // 2)
    return resample(down, len(audio))


def compression(audio):
    import numpy as np

    clipped = np.clip(audio, -0.8, 0.8)
    return np.round(clipped * 32.0) / 32.0


def unknown(audio):
    import numpy as np

    shifted = np.roll(audio, 733)
    return shifted * np.linspace(0.35, 1.65, len(shifted))


package = train_attack_ml_model(force_retrain=True)
print(package["backend"], package["validation_accuracy"])

audio, sr = load_audio("input_audio/protected.wav")

variants = {
    "clean": (audio, "PROTECTED DERIVATIVE"),
    "noise": (noise(audio), "GAUSSIAN NOISE"),
    "crop": (crop(audio), "CROPPING"),
    "lowpass": (lowpass(audio), "LOWPASS FILTER"),
    "resample": (resample_attack(audio), "RESAMPLING"),
    "gain": (gain(audio), "AMPLITUDE SCALING"),
    "compression": (compression(audio), "COMPRESSION"),
    "unknown": (unknown(audio), "UNKNOWN MODIFICATION"),
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

for name, (variant, expected_label) in variants.items():
    stft = compute_stft(variant)
    magnitude, _ = split_mag_phase(stft)
    analysis = classify_attack_ml(
        variant,
        sr,
        magnitude,
        ber=ber_values[name],
        source_hash_match=False,
        ecc_success=True,
    )

    print(name, analysis["likely_manipulation"], analysis["confidence"])
    assert analysis["likely_manipulation"] == expected_label
