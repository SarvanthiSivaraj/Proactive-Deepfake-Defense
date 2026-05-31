import librosa
import numpy as np
import soundfile as sf
import os

def test_roundtrip():
    # Create a dummy signal
    sr = 44100
    t = np.linspace(0, 5, sr * 5) # 5 seconds
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Modify one bin significantly (like QIM)
    r, c = 100, 400
    orig_val = mag[r, c]
    mag[r, c] = 5.75 # Just some value
    
    # ISTFT
    stft_mod = mag * np.exp(1j * phase)
    reconstructed = librosa.istft(stft_mod, hop_length=512)
    
    # Save and Load
    sf.write("test.wav", reconstructed, sr, subtype="FLOAT")
    loaded, _ = librosa.load("test.wav", sr=sr)
    
    # Re-STFT
    stft_back = librosa.stft(loaded, n_fft=2048, hop_length=512)
    mag_back = np.abs(stft_back)
    
    print(f"Original magnitude at (100,400): {orig_val:.4f}")
    print(f"Modified magnitude at (100,400): {mag[r,c]:.4f}")
    print(f"Reconstructed magnitude at (100,400): {mag_back[r,c]:.4f}")
    print(f"Error: {np.abs(mag[r,c] - mag_back[r,c]):.4f}")

    os.remove("test.wav")

if __name__ == "__main__":
    test_roundtrip()
