import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

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

recovered=extract_payload_qim(

    wm_mag,

    groups
)

payload=np.array(
    payload
)

recovered=np.array(
    recovered
)

ber=np.mean(
    payload!=recovered
)

print(
    "\nREAL PIPELINE BER:"
)

print(
    ber
)

wrong=np.where(
    payload!=recovered
)[0]

print(
    "\nFirst wrong indices:"
)

print(
    wrong[:20]
)