import librosa
import soundfile as sf
import numpy as np


def load_audio(filepath, sr=44100):

    # Use soundfile directly to avoid automatic normalization to [-1, 1]
    audio, file_sr = sf.read(filepath)
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    # Resample if necessary
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    return audio, sr


def normalize_audio(audio):

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak


def save_audio(filepath, audio, sr):

    sf.write(
        filepath,
        audio,
        sr,
        subtype="FLOAT"
    )