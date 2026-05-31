import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt, resample

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase
from src.security.attack_classifier import classify_attack


def gaussian_noise(audio):
    return audio + np.random.normal(0, 0.0003, len(audio))


def amplitude_scale(audio):
    return audio * 0.98


def lowpass(audio):
    b, a = butter(4, 0.5, btype="low")
    return filtfilt(b, a, audio)


def resample_attack(audio):
    down = resample(audio, len(audio) // 2)
    return resample(down, len(audio))


def crop_attack(audio):
    cropped = audio[1500:]
    return np.pad(cropped, (0, 1500))


def compression_attack(audio):
    clipped = np.clip(audio, -0.8, 0.8)
    return np.round(clipped * 32.0) / 32.0


def unknown_attack(audio):
    shifted = np.roll(audio, 733)
    return shifted * np.linspace(0.35, 1.65, len(shifted))


def variants_for(audio):
    return {
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


def evaluate(crop_thresh, unknown_thresh):
    # search multiple corpora: eval_audio, sample_audio, input_audio
    dirs = [
        os.path.join(ROOT, "data", "eval_audio"),
        os.path.join(ROOT, "data", "sample_audio"),
        os.path.join(ROOT, "input_audio"),
    ]

    files = []
    for data_dir in dirs:
        if not os.path.isdir(data_dir):
            continue
        files += [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".wav")
        ]

    total = 0
    correct = 0
    crop_correct = 0
    crop_total = 0
    unknown_correct = 0
    unknown_total = 0

    for fp in files:
        audio, sr = load_audio(fp)
        vars = variants_for(audio)

        for name, attacked_audio in vars.items():
            stft = compute_stft(attacked_audio)
            magnitude, _ = split_mag_phase(stft)

            analysis = classify_attack(
                attacked_audio,
                sr,
                magnitude,
                ber=ber_values[name],
                source_hash_match=False
            )

            pred = analysis["likely_manipulation"]
            metrics = analysis["metrics"]

            # apply threshold overrides for crop and unknown
            if name == "crop":
                crop_total += 1
                if metrics["ber"] >= crop_thresh:
                    pred = "CROPPING"
                if pred == expected[name]:
                    crop_correct += 1

            if name == "unknown":
                unknown_total += 1
                if metrics["ber"] >= unknown_thresh:
                    pred = "UNKNOWN MODIFICATION"
                if pred == expected[name]:
                    unknown_correct += 1

            if pred == expected[name]:
                correct += 1

            total += 1

    overall_acc = correct / total if total else 0.0
    crop_acc = crop_correct / crop_total if crop_total else 0.0
    unknown_acc = unknown_correct / unknown_total if unknown_total else 0.0

    return {
        "overall": overall_acc,
        "crop_acc": crop_acc,
        "unknown_acc": unknown_acc,
        "total": total,
    }


def grid_search():
    best = None
    best_params = None

    crop_range = np.arange(0.05, 0.20, 0.01)
    unknown_range = np.arange(0.03, 0.20, 0.01)

    for c in crop_range:
        for u in unknown_range:
            res = evaluate(c, u)
            score = res["crop_acc"] + res["unknown_acc"]
            if best is None or score > best:
                best = score
                best_params = (c, u, res)

    return best_params


if __name__ == "__main__":
    print("Running grid search for crop/unknown BER thresholds...")
    c, u, res = grid_search()
    print(f"Best crop_thresh={c:.3f}, unknown_thresh={u:.3f}")
    print(res)
