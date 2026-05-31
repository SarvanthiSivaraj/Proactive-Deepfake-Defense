import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *

# ==========================
# LOAD AUDIO
# ==========================

audio,sr=load_audio(
    "data/sample_audio/speech.wav"
)

stft=compute_stft(
    audio
)

mag,phase=split_mag_phase(
    stft
)

# dummy payload

payload_bits=[1,0,1,1,0]*100

embedded_mag,_=embed_payload_qim(

    mag,

    payload_bits,

    seed=42
)

modified=merge_mag_phase(

    embedded_mag,

    phase
)

watermarked=inverse_stft(
    modified
)

# ==========================
# QUALITY METRICS
# ==========================

n=min(

    len(audio),

    len(watermarked)
)

audio=audio[:n]

watermarked=watermarked[:n]

mse=np.mean(

    (audio-watermarked)**2
)

signal_power=np.mean(
    audio**2
)

snr=10*np.log10(

    signal_power
    /
    mse
)

print(
    "\nQUALITY METRICS"
)

print(
    "\nMSE:"
)

print(
    mse
)

print(
    "\nSNR (dB):"
)

print(
    snr
)