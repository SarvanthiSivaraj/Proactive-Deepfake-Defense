import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *

audio,sr=load_audio(
    "data/sample_audio/speech.wav"
)

stft=compute_stft(
    audio
)

mag,phase=split_mag_phase(
    stft
)

payload=[0,1]*100

embedded_mag,groups=embed_payload_qim(

    mag,

    payload,

    seed=42
)

modified=merge_mag_phase(

    embedded_mag,

    phase
)

watermarked=inverse_stft(
    modified
)

wm_stft=compute_stft(
    watermarked
)

wm_mag,_=split_mag_phase(
    wm_stft
)

zero=[]
one=[]

for bit,group in zip(

        payload,

        groups
):

    for row,col in group:

        rem=np.mod(

            wm_mag[
                row,
                col
            ],

            0.5
        )

        if bit==0:

            zero.append(
                rem
            )

        else:

            one.append(
                rem
            )

print(
    "\n0-bit mean:",
    np.mean(zero)
)

print(
    "0-bit std:",
    np.std(zero)
)

print(
    "\n1-bit mean:",
    np.mean(one)
)

print(
    "1-bit std:",
    np.std(one)
)