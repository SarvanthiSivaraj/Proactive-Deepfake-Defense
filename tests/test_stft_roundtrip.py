import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

audio,sr=load_audio(
    "data/eval_audio/speech2.wav"
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

diff=np.mean(
    np.abs(
        mag-rmag
    )
)

reldiff=np.mean(

    np.abs(
        mag-rmag
    )
    /
    (
        np.abs(mag)+1e-8
    )
)

print(
    "\nPURE STFT ROUNDTRIP"
)

print(
    "Absolute diff:",
    diff
)

print(
    "Relative diff:",
    reldiff
)