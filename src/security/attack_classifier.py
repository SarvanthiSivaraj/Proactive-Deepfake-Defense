import numpy as np


def _compute_attack_metrics(audio, sr, magnitude, ber):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )

    peak = float(
        np.max(
            np.abs(
                audio
            )
        )
    )

    zcr = float(
        np.mean(
            np.abs(
                np.diff(
                    np.signbit(
                        audio
                    ).astype(int)
                )
            )
        )
    )

    power = magnitude ** 2
    total_power = float(
        np.sum(
            power
        )
        + 1e-12
    )

    freqs = np.linspace(
        0.0,
        sr / 2.0,
        power.shape[0]
    )

    centroid = float(
        np.sum(
            power * freqs[:, None]
        )
        / total_power
    )

    high_band = freqs >= (0.35 * (sr / 2.0))
    low_band = freqs <= (0.15 * (sr / 2.0))

    hf_ratio = float(
        np.sum(
            power[high_band]
        )
        / total_power
    )

    lf_ratio = float(
        np.sum(
            power[low_band]
        )
        / total_power
    )

    mean_mag = np.mean(
        magnitude,
        axis=1
    )

    flatness = float(
        np.exp(
            np.mean(
                np.log(
                    mean_mag + 1e-12
                )
            )
        )
        / (
            np.mean(
                mean_mag
            )
            + 1e-12
        )
    )

    quarter = max(
        sr // 4,
        1
    )

    head = audio[:quarter]
    tail = audio[-quarter:]
    body = audio[quarter:-quarter] if len(audio) > (2 * quarter) else audio

    head_energy = float(np.mean(head ** 2))
    tail_energy = float(np.mean(tail ** 2))
    body_energy = float(np.mean(body ** 2)) + 1e-12

    edge_ratio = float(
        (head_energy + tail_energy) / body_energy
    )

    return {
        "ber": float(ber),
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "centroid": centroid,
        "hf_ratio": hf_ratio,
        "lf_ratio": lf_ratio,
        "flatness": flatness,
        "edge_ratio": edge_ratio,
    }


def classify_attack(audio, sr, magnitude, ber, source_hash_match=None):

    metrics = _compute_attack_metrics(
        audio,
        sr,
        magnitude,
        ber
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
            "Spectral flatness elevated and zero-crossing rate increased."
        )

    if metrics["hf_ratio"] <= 0.0015 and metrics["centroid"] <= 323.0:

        add_score(
            "LOWPASS FILTER",
            5,
            "High-frequency energy dropped and the spectral centroid shifted downward."
        )

    if metrics["hf_ratio"] <= 0.0018 and metrics["flatness"] <= 0.002 and metrics["zcr"] <= 0.085:

        add_score(
            "RESAMPLING",
            4,
            "Very low spectral flatness with reduced high-frequency energy suggests resampling artifacts."
        )

    if metrics["rms"] <= 0.0556 and metrics["flatness"] >= 0.15 and metrics["flatness"] <= 0.26 and metrics["ber"] < 0.06:

        add_score(
            "AMPLITUDE SCALING",
            3,
            "Energy drift is small while the spectral shape stays close to the baseline speech profile."
        )

    if metrics["ber"] >= 0.05 and metrics["flatness"] >= 0.15 and metrics["flatness"] <= 0.26 and metrics["hf_ratio"] >= 0.0018:

        add_score(
            "CROPPING",
            3,
            "BER pressure is high but the spectral profile still resembles the original speech band shape."
        )

    if metrics["hf_ratio"] >= 0.0035 and metrics["centroid"] >= 380.0 and metrics["flatness"] >= 0.35:

        add_score(
            "COMPRESSION",
            5,
            "High spectral centroid, high flatness, and elevated high-frequency energy are consistent with codec/compression artifacts."
        )

    if metrics["ber"] < 0.05 and metrics["flatness"] < 0.09 and metrics["hf_ratio"] > 0.0016:

        add_score(
            "NONE",
            2,
            "BER is low and the broad spectral shape remains stable."
        )

    best_label = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_label]
    sorted_scores = sorted(
        scores.values(),
        reverse=True
    )

    margin = best_score - sorted_scores[1] if len(sorted_scores) > 1 else best_score

    if best_score >= 5 or (best_score >= 4 and margin >= 2):

        confidence = "HIGH"

    elif best_score >= 2 or margin >= 1:

        confidence = "MEDIUM"

    else:

        confidence = "LOW"

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