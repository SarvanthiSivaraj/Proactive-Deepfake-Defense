import sys
import os

sys.path.append(
    os.path.abspath(".")
)

from src.preprocessing.loader import (

    load_audio,

    save_audio
)

from src.preprocessing.stft import (

    compute_stft,

    split_mag_phase,

    merge_mag_phase,

    inverse_stft
)

audio, sr = load_audio(

    "data/sample_audio/speech.wav"
)

print("Audio Loaded.")

stft = compute_stft(

    audio
)

print(

    "STFT shape:",

    stft.shape
)

mag, phase = split_mag_phase(

    stft
)

print(

    "Magnitude shape:",

    mag.shape
)

print(

    "Phase shape:",

    phase.shape
)

reconstructed = inverse_stft(

    merge_mag_phase(

        mag,

        phase
    )
)

save_audio(

    "data/sample_audio/reconstructed.wav",

    reconstructed,

    sr
)

print(

    "Reconstructed file saved."
)