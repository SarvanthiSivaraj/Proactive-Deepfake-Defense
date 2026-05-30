import numpy as np


def embed_qim(

        value,

        bit,

        delta=0.01
):

    q = np.round(

        value / delta
    )

    if bit == 0:

        q = 2 * np.round(

            q / 2
        )

    else:

        q = (

            2 *

            np.round(

                q / 2
            )

        ) + 1

    embedded = q * delta

    return embedded
def embed_payload_qim(

        magnitude,

        payload_bits,

        start_row=100,

        col=50
):

    mag = magnitude.copy()

    for i, bit in enumerate(

            payload_bits
    ):

        row = start_row + i

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

    return mag