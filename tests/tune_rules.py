import os
import csv
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("."))

from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase
from src.forensic.features import extract_forensic_features


def make_variants(audio, seed=42):
    rng = np.random.default_rng(seed)

    def seeded_noise(x):
        return x + rng.normal(0, rng.uniform(0.00015, 0.0006), len(x))

    def seeded_gain(x):
        return x * rng.uniform(0.92, 1.05)

    def seeded_crop(x):
        crop_len = int(rng.integers(800, min(4000, max(801, len(x) // 3))))
        cropped = x[crop_len:]
        return np.pad(cropped, (0, crop_len))[: len(x)]

    def seeded_lowpass(x):
        from scipy.signal import butter, filtfilt

        cutoff = float(rng.uniform(0.30, 0.65))
        b, a = butter(4, cutoff, btype="low")
        return filtfilt(b, a, x)

    def seeded_resample(x):
        from scipy.signal import resample

        factor = float(rng.uniform(0.55, 0.85))
        down = resample(x, max(8, int(len(x) * factor)))
        return resample(down, len(x))

    def seeded_compression(x):
        clip_level = float(rng.uniform(0.65, 0.95))
        quant = float(rng.choice([16.0, 24.0, 32.0, 48.0]))
        clipped = np.clip(x, -clip_level, clip_level)
        return np.round(clipped * quant) / quant

    def seeded_unknown(x):
        shift = int(rng.integers(256, 1400))
        shifted = np.roll(x, shift)
        envelope = np.linspace(float(rng.uniform(0.25, 0.5)), float(rng.uniform(1.2, 1.8)), len(shifted))
        return shifted * envelope

    return {
        "clean": (audio, "PROTECTED DERIVATIVE", 0.035, None),
        "noise": (seeded_noise(audio), "GAUSSIAN NOISE", 0.08, None),
        "crop": (seeded_crop(audio), "CROPPING", 0.14, None),
        "lowpass": (seeded_lowpass(audio), "LOWPASS FILTER", 0.16, None),
        "resample": (seeded_resample(audio), "RESAMPLING", 0.12, None),
        "gain": (seeded_gain(audio), "AMPLITUDE SCALING", 0.04, None),
        "compression": (seeded_compression(audio), "COMPRESSION", 0.11, None),
        "unknown": (seeded_unknown(audio), "UNKNOWN MODIFICATION", 0.09, None),
    }


def compute_metrics_for_variants(variants, sr):
    rows = []
    for name, (variant, label, ber, _) in variants.items():
        stft = compute_stft(variant)
        magnitude, _ = split_mag_phase(stft)
        metrics = extract_forensic_features(variant, sr, magnitude, ber, ecc_success=True)
        row = {"attack": name, "expected": label}
        row.update(metrics)
        rows.append(row)
    return rows


def rule_predict(metrics, params):
    # params: dict with thresholds
    # Lowpass
    if (
        metrics["hf_ratio"] <= params["lowpass_hf"]
        and metrics["centroid"] <= params["lowpass_centroid"]
        and metrics["flatness"] <= params["lowpass_flatness"]
    ):
        return "LOWPASS FILTER"

    # Resampling
    if (
        metrics["hf_ratio"] <= params["res_hf"]
        and metrics["flatness"] <= params["res_flat"]
        and metrics["zcr"] <= params["res_zcr"]
    ):
        return "RESAMPLING"

    # Cropping
    if (
        metrics["ber"] >= params["crop_ber"]
        and metrics["edge_ratio"] >= params["crop_edge"]
        and metrics["column_shift_pressure"] >= params["crop_col"]
        and metrics["hf_ratio"] >= params["crop_hf"]
        and metrics["peak"] >= params["crop_peak"]
    ):
        return "CROPPING"

    # Fallback: return NONE if broad-stable
    if metrics["ber"] < 0.05 and metrics["flatness"] < 0.09 and metrics["hf_ratio"] > 0.0016:
        return "PROTECTED DERIVATIVE"

    return "UNKNOWN MODIFICATION"


def grid_search(rows):
    # A smarter search that mirrors the rule-based classifier scoring
    def scoring_predict(metrics, params):
        scores = {
            "NONE": 0,
            "GAUSSIAN NOISE": 0,
            "AMPLITUDE SCALING": 0,
            "LOWPASS FILTER": 0,
            "RESAMPLING": 0,
            "CROPPING": 0,
            "COMPRESSION": 0,
            "UNKNOWN MODIFICATION": 0,
        }

        def add(label, points):
            scores[label] += points

        # Gaussian noise
        if metrics["flatness"] >= params["noise_flat"] and metrics["zcr"] >= params["noise_zcr"]:
            add("GAUSSIAN NOISE", 4)

        if metrics["flatness"] >= params["noise2_flat"] and metrics["zcr"] >= params["noise2_zcr"] and metrics["peak"] >= params["noise2_peak"]:
            add("GAUSSIAN NOISE", 2)

        # Lowpass
        if metrics["hf_ratio"] <= params["lowpass_hf"] and metrics["centroid"] <= params["lowpass_centroid"] and metrics["flatness"] <= params["lowpass_flatness"]:
            add("LOWPASS FILTER", 5)

        # Resampling
        if metrics["hf_ratio"] <= params["res_hf"] and metrics["flatness"] <= params["res_flat"] and metrics["zcr"] <= params["res_zcr"]:
            add("RESAMPLING", 4)

        # Amplitude scaling
        if metrics["rms"] <= params["rms_thresh"] and params["flat_low"] <= metrics["flatness"] <= params["flat_high"] and metrics["ber"] < params["ber_thresh"]:
            add("AMPLITUDE SCALING", 3)

        # Cropping
        if (
            metrics["ber"] >= params["crop_ber"]
            and metrics["edge_ratio"] >= params["crop_edge"]
            and metrics["column_shift_pressure"] >= params["crop_col"]
            and metrics["hf_ratio"] >= params["crop_hf"]
            and metrics["peak"] >= params["crop_peak"]
        ):
            add("CROPPING", 4)

        # Compression
        if metrics["hf_ratio"] >= params["comp_hf"] and metrics["centroid"] >= params["comp_centroid"] and metrics["flatness"] >= params["comp_flat"]:
            add("COMPRESSION", 5)

        # NONE
        if metrics["ber"] < params["none_ber"] and metrics["flatness"] < params["none_flat"] and metrics["hf_ratio"] > params["none_hf"]:
            add("NONE", 2)

        best_label = max(scores, key=scores.get)
        if best_label == "NONE":
            # map NONE -> PROTECTED DERIVATIVE for comparison
            return "PROTECTED DERIVATIVE"
        return best_label

    # parameter ranges (small grid)
    best = {"score": -1}
    for lowpass_hf in [0.0015, 0.002, 0.0025]:
        for lowpass_cent in [320.0, 340.0, 360.0]:
            for lowpass_flat in [0.002, 0.01, 0.05]:
                for res_hf in [0.0015, 0.002, 0.0025]:
                    for res_flat in [0.002, 0.01, 0.02]:
                        for res_zcr in [0.09, 0.1, 0.11]:
                            params = {
                                "noise_flat": 0.09,
                                "noise_zcr": 0.1,
                                "noise2_flat": 0.2,
                                "noise2_zcr": 0.1,
                                "noise2_peak": 1.005,
                                "lowpass_hf": float(lowpass_hf),
                                "lowpass_centroid": float(lowpass_cent),
                                "lowpass_flatness": float(lowpass_flat),
                                "res_hf": float(res_hf),
                                "res_flat": float(res_flat),
                                "res_zcr": float(res_zcr),
                                "rms_thresh": 0.0556,
                                "flat_low": 0.15,
                                "flat_high": 0.26,
                                "ber_thresh": 0.06,
                                "crop_ber": 0.05,
                                "crop_edge": 0.0004,
                                "crop_col": 0.02,
                                "crop_hf": 0.0018,
                                "crop_peak": 0.95,
                                "comp_hf": 0.0035,
                                "comp_centroid": 380.0,
                                "comp_flat": 0.35,
                                "none_ber": 0.05,
                                "none_flat": 0.09,
                                "none_hf": 0.0016,
                            }

                            correct = 0
                            total = 0
                            for r in rows:
                                pred = scoring_predict(r, params)
                                if pred == r["expected"]:
                                    correct += 1
                                total += 1
                            score = correct / total
                            if score > best["score"]:
                                best = {"score": score, "params": params}
    return best


def main():
    audio, sr = load_audio("input_audio/protected.wav")
    variants = make_variants(audio, seed=42)
    rows = compute_metrics_for_variants(variants, sr)

    os.makedirs("metadata", exist_ok=True)
    csv_path = os.path.join("metadata", "variant_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "expected"] + list(rows[0].keys()))
        # ensure header
        writer.fieldnames = list(rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Saved per-variant metrics to metadata/variant_metrics.csv")

    print("Running grid search for rule thresholds (may take a minute)...")
    best = grid_search(rows)
    print(f"Best rule accuracy: {best['score']:.3f}")
    tune_path = os.path.join("metadata", "rule_tuning.json")
    with open(tune_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Saved best params to {tune_path}")


if __name__ == "__main__":
    main()
