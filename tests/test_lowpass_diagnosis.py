import sys
import os

sys.path.append(
    os.path.abspath(".")
)

import numpy as np

from scipy.signal import butter,filtfilt

from src.preprocessing.loader import *
from src.preprocessing.stft import *

# ==========================
# LOAD AUDIO
# ==========================

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
    "\nORIGINAL MAG SHAPE:"
)

print(
    mag.shape
)

# ==========================
# ORIGINAL ENERGY
# ==========================

energy=np.mean(

    np.abs(mag),

    axis=1
)

print(
    "\nBIN ENERGY PROFILE"
)

for r in range(

        0,

        min(
            300,
            len(energy)
        ),

        20
):

    print(

        "row",

        r,

        "energy",

        round(

            float(
                energy[r]
            ),

            4
        )
    )

# ==========================
# LOWPASS ATTACK
# ==========================

b,a=butter(

    4,

    0.25,

    btype="low"
)

attacked=filtfilt(

    b,
    a,
    audio
)

atk_stft=compute_stft(
    attacked
)

atk_mag,_=split_mag_phase(
    atk_stft
)

atk_energy=np.mean(

    np.abs(atk_mag),

    axis=1
)

# ==========================
# SURVIVAL ANALYSIS
# ==========================

print(
    "\nLOWPASS SURVIVAL RATIOS"
)

for r in range(

        0,

        min(
            300,
            len(energy)
        ),

        20
):

    ratio=(

        atk_energy[r]

        /

        (

            energy[r]
            +
            1e-9
        )
    )

    print(

        "row",

        r,

        "ratio",

        round(

            float(
                ratio
            ),

            3
        )
    )