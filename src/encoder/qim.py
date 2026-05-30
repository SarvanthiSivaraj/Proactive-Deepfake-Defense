import numpy as np


def adaptive_delta(value):

    if value > 1.0:
        return 0.50

    elif value > 0.1:
        return 0.10

    else:
        return 0.05


def embed_qim(value, bit):

    delta = adaptive_delta(value)

    q = np.floor(
        value / delta
    )

    if bit == 0:

        return q*delta

    else:

        return (
            q*delta
            +
            delta/2
        )


def embed_payload_qim(

        magnitude,

        payload_bits
):

    mag = magnitude.copy()

    rows=[]

    base_col=200
    col_step=10

    REP=3

    for i, bit in enumerate(

            payload_bits
    ):

        for rep in range(REP):

            col = (

                base_col

                +

                i*col_step

                +

                rep*3
            )

            if col >= mag.shape[1]:

                continue

            energies = mag[:,col]

            valid_rows=np.arange(

                50,

                min(
                    500,
                    len(energies)
                )
            )

            row = valid_rows[

                np.argmax(
                    energies[
                        valid_rows
                    ]
                )
            ]

            rows.append(

                (row,col)
            )

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

    return mag, rows