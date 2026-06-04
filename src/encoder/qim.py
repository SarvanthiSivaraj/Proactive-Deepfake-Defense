import numpy as np
import pickle
import os

from src.sync.location_map import *

DELTA = 0.5
REPEAT = 9


def embed_qim(

        value,

        bit
):

    remainder = np.mod(

        value,

        DELTA
    )

    base = value - remainder

    if bit == 0:

        target = 0.05

    else:

        target = 0.45

    embedded = base + target

    if bit == 1:

        embedded += 0.15

    return embedded


def embed_payload_qim(

        magnitude,

        payload_bits,

        seed=42
):

    mag = magnitude.copy()

    requested_locations = len(payload_bits) * REPEAT
    locations = generate_location_map(

        magnitude,

        requested_locations,

        seed
    )

    if len(locations) < len(payload_bits):
        raise ValueError(
            f"Not enough QIM locations for payload: need at least {len(payload_bits)} locations, got {len(locations)}."
        )

    repeat_count = min(REPEAT, max(1, len(locations) // len(payload_bits)))
    usable_locations = len(payload_bits) * repeat_count
    locations = locations[:usable_locations]

    grouped = []

    idx = 0

    for bit in payload_bits:

        group = []

        for _ in range(repeat_count):

            row, col = locations[idx]

            idx += 1

            mag[
                row,
                col
            ] = embed_qim(

                mag[
                    row,
                    col
                ],

                bit
            )

            group.append(
                (
                    row,
                    col
                )
            )

        grouped.append(
            group
        )

    os.makedirs(
        "metadata",
        exist_ok=True
    )

    with open(
        "metadata/grouped_locations.pkl",
        "wb"
    ) as f:

        pickle.dump(
            grouped,
            f
        )

    with open(
        "metadata/payload_bits.pkl",
        "wb"
    ) as f:

        pickle.dump(
            payload_bits,
            f
        )

    return mag, grouped
