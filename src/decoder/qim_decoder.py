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


def extract_qim(

        value
):

    delta = adaptive_delta(

        value
    )

    remainder = np.mod(

        value,

        delta
    )

    dist0 = abs(

        remainder-0
    )

    dist1 = abs(

        remainder-delta/2
    )

    if dist0 < dist1:

        return 0

    else:

        return 1


def extract_payload_qim(

        magnitude,

        rows,

        col=100
):

    bits=[]

    for row in rows:

        value = magnitude[

            row,

            col
        ]

        bits.append(

            extract_qim(

                value
            )
        )

    return bits