import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

folder="data/eval_audio"

files=[

    f for f in os.listdir(folder)

    if f.endswith(".wav")
]

print(
    "\nDATASET DIAGNOSIS"
)

for filename in files:

    print(
        "\n==================="
    )

    print(
        filename
    )

    print(
        "==================="
    )

    audio,sr=load_audio(

        os.path.join(
            folder,
            filename
        )
    )

    stft=compute_stft(
        audio
    )

    mag,_=split_mag_phase(
        stft
    )

    energy=np.mean(

        np.abs(mag),

        axis=1
    )

    ranked=np.argsort(
        energy
    )[::-1]

    print(
        "TOP ENERGY ROWS:"
    )

    print(
        ranked[:20]
    )

    print(
        "TOP ENERGIES:"
    )

    print(
        np.round(
            energy[
                ranked[:20]
            ],
            4
        )
    )