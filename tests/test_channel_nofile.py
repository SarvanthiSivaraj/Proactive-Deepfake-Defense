import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import *
from src.preprocessing.stft import *

from src.encoder.qim import *
from src.decoder.qim_decoder import *

from src.evaluation.metrics import *


bits = [

    1,0,1,1,
    0,1,0,0,
    1,1,0,0,
    1,0,1,0
]

print("\nOriginal Bits:")
print(bits)


audio, sr = load_audio(

    "data/sample_audio/speech.wav"
)

stft = compute_stft(audio)

mag, phase = split_mag_phase(stft)


embedded_mag, locations = embed_payload_qim(

    mag,

    bits
)

modified = merge_mag_phase(

    embedded_mag,

    phase
)

watermarked = inverse_stft(

    modified
)

# NO SAVE / NO RELOAD

wm_stft = compute_stft(

    watermarked
)

wm_mag, wm_phase = split_mag_phase(

    wm_stft
)

recovered = extract_payload_qim(

    wm_mag,

    locations
)

print("\nRecovered Bits:")
print(recovered)

ber = compute_ber(

    bits,

    recovered
)

print("\nBER:")
print(ber)