import sys
import os

sys.path.append(
    os.path.abspath(".")
)

import numpy as np

from src.preprocessing.loader import *
from src.preprocessing.stft import *

audio,sr=load_audio(
    "data/sample_audio/speech.wav"
)

stft=compute_stft(
    audio
)

mag,_=split_mag_phase(
    stft
)

print(
    "\nORIGINAL SHAPE:"
)

print(
    mag.shape
)

cropped=audio[
    1000:
]

cropped=np.pad(

    cropped,

    (0,1000)
)

crop_stft=compute_stft(
    cropped
)

crop_mag,_=split_mag_phase(
    crop_stft
)

print(
    "\nCROP SHAPE:"
)

print(
    crop_mag.shape
)

corr=[]

for shift in range(

        -50,

        51
):

    score=0

    for r in range(

            50,

            150
    ):

        try:

            a=mag[
                r,
                200:600
            ]

            b=crop_mag[
                r,
                200+shift:
                600+shift
            ]

            n=min(
                len(a),
                len(b)
            )

            score += np.corrcoef(

                a[:n],

                b[:n]

            )[0,1]

        except:

            pass

    corr.append(
        score
    )

best_shift=np.argmax(
    corr
)-50

print(
    "\nBEST SHIFT:"
)

print(
    best_shift
)