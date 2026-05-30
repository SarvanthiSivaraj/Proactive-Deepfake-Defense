import numpy as np

DELTA = 0.5

THRESHOLD = 0.14


def extract_qim(value):

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

    bits=[]

    for group in grouped_locations:

        votes=[]

        for row,col in group:

            votes.append(

                extract_qim(

                    magnitude[
                        row,
                        col
                    ]
                )
            )

        bits.append(

            int(
                np.mean(
                    votes
                ) >= 0.5
            )
        )

    return bits