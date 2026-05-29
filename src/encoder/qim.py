import numpy as np


def embed_qim(

        value,

        bit,

        delta=0.1
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