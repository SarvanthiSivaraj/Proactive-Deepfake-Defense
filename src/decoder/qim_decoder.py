import numpy as np

DELTA = 0.5
THRESHOLD = 0.18
GROUP_TRIM = 7
GROUP_THRESHOLD = 0.165


def extract_qim(

        value
):

    remainder = np.mod(

        value,

        DELTA
    )

    if remainder < THRESHOLD:

        return 0

    else:

        return 1


def extract_payload_qim(

        magnitude,

        grouped_locations
):

    bits = []
    n_rows, n_cols = magnitude.shape[:2]

    for group in grouped_locations:
        residues = []

        for row, col in group:
            # ensure indices are integers
            try:
                r = int(row)
                c = int(col)
            except Exception:
                continue

            # skip out-of-bounds coordinates
            if r < 0 or c < 0 or r >= n_rows or c >= n_cols:
                continue

            residues.append(np.mod(magnitude[r, c], DELTA))

        if len(residues) == 0:
            # If no valid residues for this group, treat as 0 (safe default)
            bits.append(0)
            continue

        # use the smallest GROUP_TRIM residues (or all if fewer available)
        trimmed = np.sort(residues)[:GROUP_TRIM]
        bit = int(np.mean(trimmed) >= GROUP_THRESHOLD)
        bits.append(bit)

    return bits
