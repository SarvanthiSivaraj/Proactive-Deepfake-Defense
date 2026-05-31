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

    for group in grouped_locations:

        residues = []

        for row, col in group:

            residues.append(

                np.mod(

                    magnitude[
                        row,
                        col
                    ],

                    DELTA
                )
            )

        bits.append(

            int(

                np.mean(

                    np.sort(

                        residues
                    )[:GROUP_TRIM]

                ) >= GROUP_THRESHOLD

            )
        )

    return bits
