import numpy as np


def find_sync(

        signal,

        sync_code
):

    correlation = np.correlate(

        signal,

        sync_code,

        mode="valid"
    )

    best_index = np.argmax(

        correlation
    )

    confidence = np.max(

        correlation
    )

    return (

        best_index,

        confidence
    )