import os
import sys

sys.path.append(os.path.abspath("."))

from src.forensic.classifier import classify_attack
from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase


def noise(audio):
    import numpy as np

    return audio + np.random.normal(0, 0.0003, len(audio))


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


def run():
    audio, sr = load_audio("input_audio/protected.wav")
    variants = {
        "crop": crop(audio),
        "lowpass": lowpass(audio),
        "resample": resample_attack(audio),
    }

    ber_values = {"crop": 0.14, "lowpass": 0.16, "resample": 0.12}

    for name, variant in variants.items():
        stft = compute_stft(variant)
        mag, _ = split_mag_phase(stft)
        analysis = classify_attack(variant, sr, mag, ber_values[name], source_hash_match=False, ecc_success=True)
        print("===", name)
        print("Likely:", analysis["likely_manipulation"])
        print("Confidence:", analysis["confidence"])
        print("Scores:")
        for k, v in analysis["scores"].items():
            print(f"  {k}: {v}")
        print("Metrics:")
        for mk, mv in analysis["metrics"].items():
            print(f"  {mk}: {mv}")
        print()


if __name__ == "__main__":
    run()
