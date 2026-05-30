import numpy as np

from src.sync.location_map import *

DELTA = 0.5
REPEAT = 9


def embed_qim(value, bit):

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

    # strengthen 1-bit embedding

    if bit == 1:

        embedded += 0.15

    return embedded


def embed_payload_qim(

        magnitude,

        payload_bits,

        seed=42
):

    mag = magnitude.copy()

    locations = generate_location_map(

        magnitude,

        len(payload_bits)*REPEAT,

        seed
    )

    grouped=[]

    idx=0

    for bit in payload_bits:

        group=[]

        for _ in range(REPEAT):

            row,col = locations[idx]

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

                (row,col)
            )

        grouped.append(group)

    return mag, grouped