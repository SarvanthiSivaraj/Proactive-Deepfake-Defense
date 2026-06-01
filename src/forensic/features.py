import numpy as np


def extract_forensic_features(audio, sr, magnitude, ber, ecc_success=None):
    audio = np.asarray(audio, dtype=np.float32)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(int)))))

    power = magnitude ** 2
    total_power = float(np.sum(power) + 1e-12)
    freqs = np.linspace(0.0, sr / 2.0, power.shape[0])

    centroid = float(np.sum(power * freqs[:, None]) / total_power)

    high_band = freqs >= (0.35 * (sr / 2.0))
    low_band = freqs <= (0.15 * (sr / 2.0))

    hf_ratio = float(np.sum(power[high_band]) / total_power)
    lf_ratio = float(np.sum(power[low_band]) / total_power)

    mean_mag = np.mean(magnitude, axis=1)
    flatness = float(np.exp(np.mean(np.log(mean_mag + 1e-12))) / (np.mean(mean_mag) + 1e-12))

    quarter = max(sr // 4, 1)
    head = audio[:quarter]
    tail = audio[-quarter:]
    body = audio[quarter:-quarter] if len(audio) > (2 * quarter) else audio

    head_energy = float(np.mean(head ** 2))
    tail_energy = float(np.mean(tail ** 2))
    body_energy = float(np.mean(body ** 2)) + 1e-12

    edge_ratio = float((head_energy + tail_energy) / body_energy)

    column_shift_pressure = float(
        np.mean(np.abs(np.diff(magnitude, axis=1)))
        / (np.mean(magnitude) + 1e-12)
    )

    ecc_pressure = float(ber if ecc_success is not False else min(1.0, ber + 0.25))

    return {
        "ber": float(ber),
        "ecc_pressure": ecc_pressure,
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "centroid": centroid,
        "hf_ratio": hf_ratio,
        "lf_ratio": lf_ratio,
        "flatness": flatness,
        "edge_ratio": edge_ratio,
        "column_shift_pressure": column_shift_pressure,
    }
