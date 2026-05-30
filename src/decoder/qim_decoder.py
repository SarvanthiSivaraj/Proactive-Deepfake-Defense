import numpy as np


def extract_qim(

        value,

        delta=0.01
):

    q = np.round(

        value / delta
    )

    return int(

        q % 2
    )

def extract_payload_qim(

        magnitude,

        n_bits,

        start_row=100,

        col=50
):

    bits = []

    for i in range(

            n_bits
    ):

        row = start_row + i

        value = magnitude[

            row,

            col
        ]

        bit = extract_qim(

            value
        )

        bits.append(

            bit
        )

    return bits