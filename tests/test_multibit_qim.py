import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *


payload = [

    1,0,1,1,

    0,0,1,1
]


audio, sr = load_audio(

    "data/sample_audio/speech.wav"
)

stft = compute_stft(

    audio
)

mag, phase = split_mag_phase(

    stft
)


embedded_mag = embed_payload_qim(

    mag,

    payload
)


recovered = extract_payload_qim(

    embedded_mag,

    len(payload)
)


print(

    "\nOriginal Payload:"
)

print(payload)

print(

    "\nRecovered Payload:"
)

print(recovered)