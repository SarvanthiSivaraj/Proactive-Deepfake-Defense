# Proactive-Deepfake-Defense
Developing a cryptographic audio watermarking tool to combat AI voice cloning. Using Asymmetric Phase Coding and Ed25519 signatures, it embeds an imperceptible provenance layer achieving over 97.5% verification accuracy against MP3 compression and deepfake attacks in tens of milliseconds.

## Phase 3 Provenance
The watermark payload now carries richer provenance metadata: `id`, `generator`, `timestamp`, `creator`, `organization`, `model_version`, and `source_hash`.

Phase 3 now uses a source provenance model: embedding stores the original source audio fingerprint as `source_hash`, while verification computes `verified_audio_hash` from the file under inspection and reports both values.

The current report includes a provenance summary and a final decision such as `AUTHENTIC PROTECTED SOURCE MATCH` or `AUTHENTIC PROTECTED DERIVATIVE`.

To exercise the provenance layer without the full ECC round trip, run `tests/test_provenance_layer.py`.
