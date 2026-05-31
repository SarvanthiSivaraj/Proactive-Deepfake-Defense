import numpy as np


def generate_location_map(

        magnitude,

        n_locations,

        seed=42
):

    rng = np.random.RandomState(
        seed
    )

    rows, cols = magnitude.shape

    energy = np.mean(

        np.abs(magnitude),

        axis=1
    )

    candidate_rows=[]

    for r in range(

            30,

            min(rows,350)
    ):

        e = energy[r]

        if 0.1 < e < 10:

            candidate_rows.append(
                r
            )

    locations=[]

    for r in candidate_rows:

        for c in range(

                200,

                min(cols,900)
        ):

            locations.append(

                (int(r),int(c))
            )

    rng.shuffle(
        locations
    )

    return locations[
        :n_locations
    ]
