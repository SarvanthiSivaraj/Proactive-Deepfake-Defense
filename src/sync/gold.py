import numpy as np


def generate_gold_code(

        length=31,

        seed=42
):

    np.random.seed(

        seed
    )

    code = np.random.choice(

        [-1,1],

        size=length
    )

    return code