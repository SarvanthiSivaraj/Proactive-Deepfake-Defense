import os
import sys
import csv
import shutil
import tempfile

import numpy as np

sys.path.append(os.path.abspath("."))

from src.forensic.classifier import classify_attack as classify_attack_rules
from src.forensic.ml import (
    classify_attack_ml,
    train_attack_ml_model,
    MODEL_FILE,
)
from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase


def noise(audio):
    return audio + np.random.normal(0, 0.0003, len(audio))


def gain(audio):
    return audio * 0.98


def crop(audio):
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
    clipped = np.clip(audio, -0.8, 0.8)
    return np.round(clipped * 32.0) / 32.0


def unknown(audio):
    shifted = np.roll(audio, 733)
    return shifted * np.linspace(0.35, 1.65, len(shifted))


def make_variants(audio, seed):
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
        "clean": (audio, "PROTECTED DERIVATIVE"),
        "noise": (seeded_noise(audio), "GAUSSIAN NOISE"),
        "crop": (seeded_crop(audio), "CROPPING"),
        "lowpass": (seeded_lowpass(audio), "LOWPASS FILTER"),
        "resample": (seeded_resample(audio), "RESAMPLING"),
        "gain": (seeded_gain(audio), "AMPLITUDE SCALING"),
        "compression": (seeded_compression(audio), "COMPRESSION"),
        "unknown": (seeded_unknown(audio), "UNKNOWN MODIFICATION"),
    }


def test_rule_vs_ml_comparison():
    print("Loading or training ML model...", flush=True)
    # Ensure ML model exists (train or load)
    train_attack_ml_model(force_retrain=False)

    # Freeze baseline model once (copy if not already copied)
    baseline_path = os.path.join("metadata", "baseline_v1.joblib")
    if not os.path.exists(baseline_path) and os.path.exists(MODEL_FILE):
        shutil.copyfile(MODEL_FILE, baseline_path)

    print("Evaluating fixed attack variants...", flush=True)
    audio, sr = load_audio("input_audio/protected.wav")
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

    rows = []
    rule_correct = 0
    ml_correct = 0
    seed_scores = []

    variants = make_variants(audio, seed=42)

    for name, (variant, expected_label) in variants.items():
        stft = compute_stft(variant)
        magnitude, _ = split_mag_phase(stft)

        # Run rule classifier once
        rule_analysis = classify_attack_rules(
            variant,
            sr,
            magnitude,
            ber=ber_values[name],
            source_hash_match=False,
            ecc_success=True,
        )

        # Run ML classifier multiple times to check consistency
        ml_labels = []
        ml_confidences = []
        for _ in range(5):
            ml_analysis = classify_attack_ml(
                variant,
                sr,
                magnitude,
                ber=ber_values[name],
                source_hash_match=False,
                ecc_success=True,
            )
            ml_labels.append(ml_analysis["likely_manipulation"])
            ml_confidences.append(ml_analysis["confidence"])

        ml_label = ml_labels[0]
        ml_confidence = ml_confidences[0]

        rule_label = rule_analysis["likely_manipulation"]
        rule_confidence = rule_analysis.get("confidence")

        rule_consistent = all(
            rule_label
            == classify_attack_rules(
                variant,
                sr,
                magnitude,
                ber=ber_values[name],
                source_hash_match=False,
                ecc_success=True,
            )["likely_manipulation"]
            for _ in range(3)
        )

        ml_consistent = all(l == ml_label for l in ml_labels)

        if rule_label == expected_label:
            rule_correct += 1
        if ml_label == expected_label:
            ml_correct += 1

        rows.append(
            {
                "attack": name,
                "expected": expected_label,
                "rule_label": rule_label,
                "rule_confidence": rule_confidence,
                "rule_consistent": rule_consistent,
                "ml_label": ml_label,
                "ml_confidence": ml_confidence,
                "ml_consistent": ml_consistent,
            }
        )

    # Repeated-seed sanity check: retrain on the same source with different seeds
    # and confirm the ML backend stays stable on the same evaluation examples.
    print("Running repeated-seed stability checks...", flush=True)
    for seed in [7, 21, 42, 77, 101]:
        print(f"  Seed {seed}...", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = os.path.join(tmpdir, f"attack_ml_{seed}.joblib")
            train_attack_ml_model(force_retrain=True, model_path=tmp_model, random_state=seed)
            seed_variants = make_variants(audio, seed=seed)

            matches = 0
            total = 0
            for attack_name, (variant, expected_label) in seed_variants.items():
                stft = compute_stft(variant)
                magnitude, _ = split_mag_phase(stft)
                ml_analysis = classify_attack_ml(
                    variant,
                    sr,
                    magnitude,
                    ber=ber_values[attack_name],
                    source_hash_match=False,
                    ecc_success=True,
                    model_path=tmp_model,
                )
                total += 1
                if ml_analysis["likely_manipulation"] == expected_label:
                    matches += 1
            seed_scores.append(matches / total)

    # Write CSV artifact
    os.makedirs("metadata", exist_ok=True)
    csv_path = os.path.join("metadata", "rule_vs_ml.csv")
    print("Writing comparison CSV artifact...", flush=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack",
                "expected",
                "rule_label",
                "rule_confidence",
                "rule_consistent",
                "ml_label",
                "ml_confidence",
                "ml_consistent",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    total = len(variants)
    rule_accuracy = rule_correct / total
    ml_accuracy = ml_correct / total

    print(f"Rule accuracy: {rule_accuracy:.3f}", flush=True)
    print(f"ML accuracy:   {ml_accuracy:.3f}", flush=True)
    seed_stability = sum(seed_scores) / len(seed_scores)
    print(f"Seed stability: {seed_stability:.3f}", flush=True)

    print(flush=True)
    print("Artifacts:", flush=True)
    print(" - CSV: metadata/rule_vs_ml.csv", flush=True)
    print(" - Baseline model: metadata/baseline_v1.joblib", flush=True)

    # Requirements for Phase 4B validation:
    # - ML classifier must perform well and be stable across seeds
    # - Rule-based classifier may be imperfect; report results for manual tuning
    assert ml_accuracy >= 0.95
    assert seed_stability >= 0.95
    # Report rule accuracy but don't fail the run if rules are imperfect
    if rule_accuracy < 1.0:
        print("Note: rule-based classifier did not reach perfect accuracy; see CSV for details.", flush=True)


if __name__ == "__main__":
    test_rule_vs_ml_comparison()
