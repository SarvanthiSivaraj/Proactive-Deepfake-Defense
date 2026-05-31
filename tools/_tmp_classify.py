import sys, os
ROOT = r'C:\Users\srsar\Desktop\projects\Proactive-Deepfake-Defense'
sys.path.append(ROOT)
from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase
from src.security.attack_classifier import classify_attack
import numpy as np

audio_path = os.path.join('data', 'eval_audio', 'speech1.wav')
audio, sr = load_audio(audio_path)
cropped = np.pad(audio[1500:], (0,1500))
stft = compute_stft(cropped)
magnitude, _ = split_mag_phase(stft)
analysis = classify_attack(cropped, sr, magnitude, ber=0.14, source_hash_match=False)
print('label:', analysis['likely_manipulation'])
print('confidence:', analysis['confidence'])
print('metrics:', {k: round(v,6) for k,v in analysis['metrics'].items()})
