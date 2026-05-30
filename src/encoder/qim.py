import numpy as np


def adaptive_delta(

        value
):

    if value > 1.0:

        return 0.5

    elif value > 0.1:

        return 0.10

    else:

        return 0.05


def embed_qim(

        value,

        bit
):

    delta = adaptive_delta(

        value
    )

    q = np.floor(

        value / delta
    )

    if bit == 0:

        embedded = q*delta

    else:

        embedded = (

            q*delta

            +

            delta/2
        )

    return embedded


def embed_payload_qim(

        magnitude,

        payload_bits,

        col=100
):

    mag = magnitude.copy()

    energies = mag[:,col]

    strongest_rows = (

        energies.argsort()

        [-len(payload_bits):]

    )

    strongest_rows = sorted(

        strongest_rows
    )

    for row, bit in zip(

            strongest_rows,

            payload_bits
    ):

        original = mag[

            row,

            col
        ]

        mag[

            row,

            col

        ] = embed_qim(

            original,

            bit
        )

    return mag, strongest_rows