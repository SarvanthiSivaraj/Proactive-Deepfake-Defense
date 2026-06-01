import os
import pickle
from datetime import datetime

import joblib
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase

from .features import extract_forensic_features

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


MODEL_DIR = "metadata"
MODEL_FILE = os.path.join(MODEL_DIR, "forensic_ml_model.joblib")

FEATURE_COLUMNS = [
    "ber",
    "ecc_pressure",
    "rms",
    "peak",
    "zcr",
    "centroid",
    "hf_ratio",
    "lf_ratio",
    "flatness",
    "edge_ratio",
    "column_shift_pressure",
]

LABEL_MAP = {
    "clean": "PROTECTED DERIVATIVE",
    "noise": "GAUSSIAN NOISE",
    "crop": "CROPPING",
    "lowpass": "LOWPASS FILTER",
    "resample": "RESAMPLING",
    "gain": "AMPLITUDE SCALING",
    "compression": "COMPRESSION",
    "unknown": "UNKNOWN MODIFICATION",
}


def _collect_audio_files():
    candidates = []
    for folder in ["input_audio", "data/eval_audio", "data/sample_audio"]:
        if not os.path.isdir(folder):
            continue
        for entry in os.listdir(folder):
            if entry.lower().endswith(".wav"):
                path = os.path.join(folder, entry)
                if path not in candidates:
                    candidates.append(path)
    return candidates


def _gaussian_noise(audio, rng):
    return audio + rng.normal(0, 0.0003, len(audio))


def _amplitude_scale(audio, _rng):
    return audio * 0.98


def _lowpass(audio, _rng):
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 0.5, btype="low")
    return filtfilt(b, a, audio)


def _resample_attack(audio, _rng):
    from scipy.signal import resample

    down = resample(audio, len(audio) // 2)
    return resample(down, len(audio))


def _crop_attack(audio, _rng):
    cropped = audio[1500:]
    return np.pad(cropped, (0, 1500))


def _compression_attack(audio, _rng):
    clipped = np.clip(audio, -0.8, 0.8)
    return np.round(clipped * 32.0) / 32.0


def _unknown_attack(audio, _rng):
    shifted = np.roll(audio, 733)
    return shifted * np.linspace(0.35, 1.65, len(shifted))


VARIANT_BUILDERS = {
    "clean": lambda audio, rng: audio,
    "noise": _gaussian_noise,
    "crop": _crop_attack,
    "lowpass": _lowpass,
    "resample": _resample_attack,
    "gain": _amplitude_scale,
    "compression": _compression_attack,
    "unknown": _unknown_attack,
}


def _feature_vector(metrics):
    return np.array([metrics[column] for column in FEATURE_COLUMNS], dtype=np.float32)


def _extract_features_for_audio(audio, sr, attack_name, ber):
    stft = compute_stft(audio)
    magnitude, _ = split_mag_phase(stft)
    features = extract_forensic_features(audio, sr, magnitude, ber, ecc_success=True)
    return _feature_vector(features)


def build_attack_dataset(random_state=42):
    features = []
    labels = []

    audio_files = _collect_audio_files()
    if not audio_files:
        raise RuntimeError("No audio files were found to build the ML dataset.")

    rng = np.random.default_rng(random_state)

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

    for audio_file in audio_files:
        audio, sr = load_audio(audio_file)
        for attack_name, builder in VARIANT_BUILDERS.items():
            variant = builder(audio, rng)
            stft = compute_stft(variant)
            magnitude, _ = split_mag_phase(stft)
            metrics = extract_forensic_features(
                variant,
                sr,
                magnitude,
                ber_values[attack_name],
                ecc_success=True,
            )
            features.append(_feature_vector(metrics))
            labels.append(LABEL_MAP[attack_name])

    return np.vstack(features), np.array(labels, dtype=object)


def _candidate_models(random_state=42):
    models = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
            ),
        ),
    ]

    if XGBClassifier is not None:
        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=random_state,
                    tree_method="hist",
                ),
            )
        )

    if LGBMClassifier is not None:
        models.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.08,
                    objective="multiclass",
                    random_state=random_state,
                    min_child_samples=1,
                    num_leaves=31,
                    verbosity=-1,
                ),
            )
        )

    return models


def train_attack_ml_model(force_retrain=False, model_path=MODEL_FILE, random_state=42):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(model_path) and not force_retrain:
        return joblib.load(model_path)

    X, y = build_attack_dataset(random_state=random_state)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    stratify = y_encoded if len(np.unique(y_encoded)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )

    best_package = None
    best_accuracy = -1.0

    for backend, model in _candidate_models(random_state=random_state):
        try:
            # Some model wrappers (e.g., LightGBM sklearn wrapper) may emit a
            # UserWarning when predicting with a numpy array if the fitted
            # model stored feature names. Silence that specific warning here
            # during training/evaluation to avoid noisy output.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names",
                    category=UserWarning,
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
        except Exception:
            continue

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_package = {
                "backend": backend,
                "model": model,
                "label_encoder": encoder,
                "feature_columns": FEATURE_COLUMNS,
                "validation_accuracy": float(accuracy),
                "trained_at": datetime.utcnow().isoformat() + "Z",
            }

    if best_package is None:
        raise RuntimeError("No ML attack classifier could be trained.")

    joblib.dump(best_package, model_path)
    return best_package


def load_attack_ml_model(model_path=MODEL_FILE):
    if not os.path.exists(model_path):
        return train_attack_ml_model(model_path=model_path)
    return joblib.load(model_path)


def _confidence_from_proba(probabilities):
    sorted_probs = np.sort(probabilities)[::-1]
    top = float(sorted_probs[0])
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top - second

    if top >= 0.75 or (top >= 0.65 and margin >= 0.2):
        return "HIGH"

    if top >= 0.45 or margin >= 0.1:
        return "MEDIUM"

    return "LOW"


def classify_attack_ml(audio, sr, magnitude, ber, source_hash_match=None, ecc_success=None, model_path=MODEL_FILE):
    package = load_attack_ml_model(model_path=model_path)
    metrics = extract_forensic_features(audio, sr, magnitude, ber, ecc_success=ecc_success)
    vector = _feature_vector(metrics).reshape(1, -1)

    model = package["model"]
    encoder = package["label_encoder"]

    # LightGBM (and some sklearn wrappers) may raise a UserWarning when the
    # prediction input is a raw numpy array but the fitted model stored
    # feature names. Silence only that specific warning here to avoid
    # noisy output during CLI/test runs.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        probabilities = model.predict_proba(vector)[0]
        encoded_prediction = int(model.predict(vector)[0])
    predicted_label = encoder.inverse_transform([encoded_prediction])[0]

    score_map = {
        encoder.inverse_transform([int(cls)])[0]: float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    confidence = _confidence_from_proba(probabilities)

    return {
        "likely_manipulation": predicted_label,
        "confidence": confidence,
        "evidence": [
            f"ML backend: {package['backend']}",
            f"Validation accuracy: {package['validation_accuracy']:.3f}",
        ],
        "metrics": metrics,
        "scores": score_map,
        "backend": package["backend"],
        "validation_accuracy": package["validation_accuracy"],
        "trained_at": package.get("trained_at"),
    }
