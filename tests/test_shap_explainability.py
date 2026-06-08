import os
import sys

sys.path.append(os.path.abspath("."))

import numpy as np
from src.preprocessing.loader import load_audio
from src.preprocessing.stft import compute_stft, split_mag_phase
from src.verification.service import DeepfakeDefenseService

def test_shap_explainability():
    # Instantiate the service
    service = DeepfakeDefenseService()
    
    # Load default audio file
    audio, sr = load_audio("input_audio/protected.wav")
    stft = compute_stft(audio)
    magnitude, _ = split_mag_phase(stft)
    ber = 0.02
    
    # Run explain_prediction
    explainability = service.explain_prediction(
        audio=audio,
        sr=sr,
        magnitude=magnitude,
        ber=ber,
        ecc_success=True
    )
    
    # Assertions
    assert explainability is not None
    assert "use_shap" in explainability
    assert explainability["use_shap"] is True
    
    assert "shap_values" in explainability
    shap_vals = explainability["shap_values"]
    assert len(shap_vals) == 11  # There are 11 forensic features
    
    # Validate each shap value format
    for item in shap_vals:
        assert "feature" in item
        assert "shap_value" in item
        assert isinstance(item["feature"], str)
        assert isinstance(item["shap_value"], float)
        
    assert "base_value" in explainability
    assert isinstance(explainability["base_value"], float)
    
    # Feature importance fallback check
    assert "feature_importance" in explainability
    assert len(explainability["feature_importance"]) == 11
    
    print("SHAP Explainability unit test passed successfully!")

if __name__ == "__main__":
    test_shap_explainability()
