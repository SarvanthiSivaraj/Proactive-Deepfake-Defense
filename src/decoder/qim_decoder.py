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