import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

audio,sr=load_audio(
    "data/sample_audio/speech.wav"
)

stft=compute_stft(
    audio
)

mag,phase=split_mag_phase(
    stft
)

reconstructed=inverse_stft(
    merge_mag_phase(
        mag,
        phase
    )
)

restft=compute_stft(
    reconstructed
)

rmag,_=split_mag_phase(
    restft
)

rows,cols=mag.shape

ratio=[]

for r in range(rows):

    orig=np.mean(
        mag[r]
    )

    err=np.mean(

        np.abs(

            mag[r]-rmag[r]
        )
    )

    stability=err/(orig+1e-8)

    ratio.append(
        stability
    )

ratio=np.array(
    ratio
)

best=np.argsort(
    ratio
)

print(
    "\nMOST STABLE ROWS"
)

for r in best[:40]:

    print(

        "row",

        r,

        "stability",

        round(
            ratio[r],
            5
        ),

        "energy",

        round(
            np.mean(
                mag[r]
            ),
            5
        )
    )