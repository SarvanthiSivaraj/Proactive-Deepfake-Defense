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

mag, phase = split_mag_phase(
    stft
)


embedded_mag, groups = embed_payload_qim(
    mag,
    bits
)


print("\nDEBUG EMBEDDED VALUES")

for bit_i,group in enumerate(

        groups[:16]
):

    print(

        f"\nBIT {bit_i}"
    )

    for r,c in group:

        print(

            "loc=",
            (r,c),

            "embedded=",
            embedded_mag[r,c]
        )


modified = merge_mag_phase(
    embedded_mag,
    phase
)

watermarked = inverse_stft(
    modified
)


save_audio(
    "data/sample_audio/channel_test.wav",
    watermarked,
    sr
)


wm_audio, sr = load_audio(
    "data/sample_audio/channel_test.wav"
)

wm_stft = compute_stft(
    wm_audio
)

wm_mag, wm_phase = split_mag_phase(
    wm_stft
)


print("\nDEBUG RELOADED VALUES")

for bit_i,group in enumerate(

        groups[:16]
):

    print(

        f"\nBIT {bit_i}"
    )

    for r,c in group:

        value = wm_mag[r,c]

        decoded = extract_qim(
            value
        )

        print(

            "reload=",
            value,

            "decoded=",
            decoded
        )


recovered = extract_payload_qim(
    wm_mag,
    groups
)


print("\nRecovered Bits:")
print(recovered)


ber = compute_ber(
    bits,
    recovered
)

print("\nBER:")
print(ber)