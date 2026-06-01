from .confidence import estimate_confidence
from .features import extract_forensic_features


def classify_attack(audio, sr, magnitude, ber, source_hash_match=None, ecc_success=None):
    metrics = extract_forensic_features(
        audio,
        sr,
        magnitude,
        ber,
        ecc_success=ecc_success,
    )

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

    evidence = []

    def add_score(label, points, reason):
        scores[label] += points
        evidence.append(reason)

    if metrics["flatness"] >= 0.09 and metrics["zcr"] >= 0.1:
        add_score(
            "GAUSSIAN NOISE",
            4,
            "Spectral flatness elevated and zero-crossing rate increased.",
        )

    if metrics["flatness"] >= 0.2 and metrics["zcr"] >= 0.1 and metrics["peak"] >= 1.005:
        add_score(
            "GAUSSIAN NOISE",
            2,
            "Noise-like flatness and energy disturbance reinforce the Gaussian signature.",
        )

    if metrics["hf_ratio"] <= 0.0015 and metrics["centroid"] <= 323.0:
        add_score(
            "LOWPASS FILTER",
            5,
            "High-frequency energy dropped and the spectral centroid shifted downward.",
        )

    if metrics["hf_ratio"] <= 0.0018 and metrics["flatness"] <= 0.002 and metrics["zcr"] <= 0.085:
        add_score(
            "RESAMPLING",
            4,
            "Very low spectral flatness with reduced high-frequency energy suggests resampling artifacts.",
        )

    if metrics["rms"] <= 0.0556 and 0.15 <= metrics["flatness"] <= 0.26 and metrics["ber"] < 0.06:
        add_score(
            "AMPLITUDE SCALING",
            3,
            "Energy drift is small while the spectral shape stays close to the baseline speech profile.",
        )

    if (
        metrics["ber"] >= 0.05
        and metrics["edge_ratio"] >= 0.0004
        and metrics["column_shift_pressure"] >= 0.02
        and metrics["hf_ratio"] >= 0.0018
        and metrics["peak"] >= 0.95
    ):
        add_score(
            "CROPPING",
            4,
            "Edge activity and column-shift pressure suggest a synchronization break.",
        )

    if metrics["hf_ratio"] >= 0.0035 and metrics["centroid"] >= 380.0 and metrics["flatness"] >= 0.35:
        add_score(
            "COMPRESSION",
            5,
            "High spectral centroid, high flatness, and elevated high-frequency energy are consistent with codec/compression artifacts.",
        )

    if metrics["ber"] < 0.05 and metrics["flatness"] < 0.09 and metrics["hf_ratio"] > 0.0016:
        add_score(
            "NONE",
            2,
            "BER is low and the broad spectral shape remains stable.",
        )

    best_label = max(scores, key=scores.get)
    confidence = estimate_confidence(scores)

    if best_label == "NONE":
        if source_hash_match is False and metrics["ber"] >= 0.04:
            best_label = "UNKNOWN MODIFICATION"
        elif source_hash_match is False:
            best_label = "PROTECTED DERIVATIVE"
        else:
            best_label = "AUTHENTIC ORIGINAL"

    return {
        "likely_manipulation": best_label,
        "confidence": confidence,
        "evidence": evidence,
        "metrics": metrics,
        "scores": scores,
    }
