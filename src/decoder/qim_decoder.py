import numpy as np

DELTA = 0.5
THRESHOLD = 0.145

# crop compensation
COLUMN_SHIFT = -2


def extract_qim(value):

    remainder=np.mod(
        value,
        DELTA
    )

    if remainder < THRESHOLD:

        return 0

    return 1


def extract_payload_qim(

        magnitude,
        grouped_locations
):

    bits=[]

    cols=magnitude.shape[1]

    for group in grouped_locations:

        votes=[]

        for row,col in group:

            shifted_col=col

            # apply only when safe

            if shifted_col < 0:

                continue

            if shifted_col >= cols:

                continue

            votes.append(

                extract_qim(

                    magnitude[
                        row,
                        shifted_col
                    ]
                )
            )

        if len(votes)==0:

            bits.append(0)

        else:

            bits.append(

                int(
                    np.mean(
                        votes
                    )>=0.5
                )
            )

    return bits