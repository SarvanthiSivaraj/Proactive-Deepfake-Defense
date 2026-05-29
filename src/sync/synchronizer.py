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

    peak = np.max(

        correlation
    )

    mean_corr = np.mean(

        correlation
    )

    std_corr = np.std(

        correlation
    )

    confidence = (

        peak - mean_corr

    ) / (

        std_corr + 1e-8
    )

    return {

        "position": best_index,

        "confidence": confidence,

        "peak": peak
    }