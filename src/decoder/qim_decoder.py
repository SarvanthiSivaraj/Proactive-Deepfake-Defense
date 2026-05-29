import numpy as np


def extract_qim(

        value,

        delta=0.1
):

    q = np.round(

        value / delta
    )

    return int(

        q % 2
    )