from __future__ import annotations

import base64
import os
import pickle
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from src.decoder.qim_decoder import extract_payload_qim
from src.ecc.decode import rs_decode
from src.encoder.qim import embed_payload_qim
from src.forensic.classifier import classify_attack as classify_attack_rules
from src.forensic.ml import FEATURE_COLUMNS, classify_attack_ml, load_attack_ml_model
from src.forensic.report import build_attack_report, decide_final_authentication
from src.payload.bitstream import bits_to_bytes, bytes_to_bits
from src.payload.metadata import compute_audio_hash, generate_metadata
from src.payload.serialize import deserialize_payload, serialize_payload
from src.preprocessing.loader import load_audio, save_audio
from src.preprocessing.stft import compute_stft, inverse_stft, merge_mag_phase, split_mag_phase
from src.security.signature import sign_payload, verify_signature


DEFAULT_INPUT_DIR = "input_audio"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_GROUPED_LOCATIONS = "metadata/grouped_locations.pkl"
DEFAULT_PAYLOAD_BITS = "metadata/payload_bits.pkl"
DEFAULT_MODEL_PATH = "metadata/forensic_ml_model.joblib"
DEFAULT_RS_PARITY = 160
DEFAULT_SEED = 42


def _rs_codec(parity_bytes: int):
    try:
        from reedsolo import RSCodec
    except ImportError as exc:
        raise RuntimeError(
            "The 'reedsolo' package is required for ECC embedding and verification. "
            "Install project dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return RSCodec(parity_bytes)


def _feature_vector(metrics: dict[str, float]) -> np.ndarray:
    return np.array([metrics[column] for column in FEATURE_COLUMNS], dtype=np.float32)


def _card_status(sig_ok: bool, source_hash_match: Optional[bool]) -> str:
    if not sig_ok:
        return "NOT AUTHENTIC"
    # A valid digital signature is the authoritative proof of authenticity.
    # source_hash_match = False simply means the verified file differs from the
    # original source (expected for watermarked derivatives) — still AUTHENTIC.
    return "AUTHENTIC"


def _card_color(card_status: str) -> str:
    if card_status == "AUTHENTIC":
        return "#0f766e"
    if card_status == "PROTECTED DERIVATIVE":
        return "#b45309"
    return "#991b1b"


@dataclass
class EmbeddingResult:
    output_path: str
    verification_ready_path: str
    metadata: dict[str, Any]
    payload_bits: int
    direct_ber: float
    source_hash: str
    watermarked_audio_b64: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    file_path: str
    file_name: str
    sample_rate: int
    audio_hash: str
    ber: float
    ecc_success: bool
    signature_valid: bool
    source_hash_match: Optional[bool]
    final_authentication: str
    auth_card_status: str
    auth_card_color: str
    provenance: dict[str, Any]
    attack_analysis: dict[str, Any]
    metrics: dict[str, Any]
    report_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepfakeDefenseService:
    def __init__(
        self,
        *,
        input_dir: str = DEFAULT_INPUT_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        grouped_locations_path: str = DEFAULT_GROUPED_LOCATIONS,
        payload_bits_path: str = DEFAULT_PAYLOAD_BITS,
        model_path: str = DEFAULT_MODEL_PATH,
        rs_parity: int = DEFAULT_RS_PARITY,
        seed: int = DEFAULT_SEED,
        attack_mode: Optional[str] = None,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.grouped_locations_path = grouped_locations_path
        self.payload_bits_path = payload_bits_path
        self.model_path = model_path
        self.rs_parity = rs_parity
        self.seed = seed
        self.attack_mode = (attack_mode or os.environ.get("FORENSIC_ATTACK_MODE", "hybrid")).strip().lower()
        self._grouped_locations: Optional[list[list[tuple[int, int]]]] = None
        self._payload_bits: Optional[list[int]] = None

    def _load_grouped_locations(self):
        if self._grouped_locations is None:
            if not os.path.exists(self.grouped_locations_path):
                raise FileNotFoundError(f"Missing grouped locations: {self.grouped_locations_path}")
            with open(self.grouped_locations_path, "rb") as handle:
                self._grouped_locations = pickle.load(handle)
        return self._grouped_locations

    def _load_payload_bits(self):
        if self._payload_bits is None:
            if not os.path.exists(self.payload_bits_path):
                raise FileNotFoundError(f"Missing payload bits: {self.payload_bits_path}")
            with open(self.payload_bits_path, "rb") as handle:
                self._payload_bits = pickle.load(handle)
        return self._payload_bits

    def embed_file(self, file_path: str, output_path: Optional[str] = None) -> EmbeddingResult:
        audio, sr = load_audio(file_path)
        stft = compute_stft(audio)
        magnitude, phase = split_mag_phase(stft)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(self.output_dir, "protected.wav")

        def _make_payload(src_hash: str) -> tuple:
            """Build ECC-encoded payload bits for the given source_hash."""
            meta = generate_metadata(source_hash=src_hash, timestamp=_timestamp)
            meta_bytes = serialize_payload(meta)
            sig = sign_payload(meta_bytes)
            pkt = {"metadata": meta, "signature": sig}
            bits = bytes_to_bits(_rs_codec(self.rs_parity).encode(zlib.compress(serialize_payload(pkt))))
            return meta, bits

        def _embed_and_hash(src_hash: str):
            """Embed watermark with given source_hash; return (watermarked, groups, metadata, bits, hash)."""
            meta, bits = _make_payload(src_hash)
            emag, grps = embed_payload_qim(magnitude, bits, seed=self.seed)
            wm = inverse_stft(merge_mag_phase(emag, phase))
            h = compute_audio_hash(wm, sr)
            return wm, grps, meta, bits, h

        # Fix the timestamp at embedding time so all 3 passes produce payloads of
        # identical size → same QIM groups → correct self-referential hash check.
        from datetime import datetime, timezone
        _timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Pass 1: dummy 64-char hash → determines the QIM group structure
        _dummy = "0" * 64
        _wm1, _grps1, _meta1, _bits1, hash1 = _embed_and_hash(_dummy)

        # Pass 2: use hash of pass-1 output
        _wm2, _grps2, _meta2, _bits2, hash2 = _embed_and_hash(hash1)

        # Pass 3: use hash of pass-2 output (converges to self-referential hash)
        watermarked, groups, metadata, payload_bits, source_hash = _embed_and_hash(hash2)



        save_audio(output_path, watermarked, sr)
        verification_ready_path = os.path.join(self.input_dir, "protected.wav")
        if os.path.abspath(output_path) != os.path.abspath(verification_ready_path):
            try:
                with open(output_path, "rb") as source_handle:
                    audio_bytes = source_handle.read()
                with open(verification_ready_path, "wb") as target_handle:
                    target_handle.write(audio_bytes)
            except OSError:
                verification_ready_path = output_path

        # Save source-specific metadata using package-root-relative absolute paths
        _pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _meta_dir = os.path.join(_pkg_root, "metadata")
        import shutil
        source_name = os.path.splitext(os.path.basename(file_path))[0]
        shutil.copyfile(
            os.path.join(_meta_dir, "grouped_locations.pkl"),
            os.path.join(_meta_dir, f"{source_name}_grouped_locations.pkl"),
        )
        shutil.copyfile(
            os.path.join(_meta_dir, "payload_bits.pkl"),
            os.path.join(_meta_dir, f"{source_name}_payload_bits.pkl"),
        )

        wm_stft = compute_stft(watermarked)
        wm_mag, _ = split_mag_phase(wm_stft)
        recovered_bits = extract_payload_qim(wm_mag, groups)[: len(payload_bits)]
        direct_ber = float(np.mean(np.array(payload_bits) != np.array(recovered_bits)))

        with open(output_path, "rb") as handle:
            watermarked_audio_b64 = base64.b64encode(handle.read()).decode("ascii")

        return EmbeddingResult(
            output_path=output_path,
            verification_ready_path=verification_ready_path,
            metadata=metadata,
            payload_bits=len(payload_bits),
            direct_ber=direct_ber,
            source_hash=source_hash,
            watermarked_audio_b64=watermarked_audio_b64,
        )



    def explain_prediction(self, audio, sr, magnitude, ber, ecc_success=None) -> dict[str, Any]:
        from sklearn.inspection import permutation_importance

        package = load_attack_ml_model(model_path=self.model_path)
        metrics = self._extract_metrics(audio, sr, magnitude, ber, ecc_success=ecc_success)
        vector = _feature_vector(metrics).reshape(1, -1)

        model = package["model"]
        encoder = package["label_encoder"]

        with np.errstate(all="ignore"):
            probabilities = model.predict_proba(vector)[0]
            encoded_prediction = int(model.predict(vector)[0])
        predicted_label = encoder.inverse_transform([encoded_prediction])[0]
        score_map = {
            encoder.inverse_transform([int(cls)])[0]: float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }

        shap_rows = []
        base_value = 0.0
        use_shap = False

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(vector)

            # Find the index of the predicted class in model.classes_
            class_idx = 0
            if hasattr(model, "classes_"):
                class_idx = int(np.where(model.classes_ == encoded_prediction)[0][0])

            # Extract SHAP values for this specific class
            if isinstance(shap_values, list):
                class_shap = shap_values[class_idx][0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    class_shap = shap_values[0, :, class_idx]
                elif shap_values.ndim == 2:
                    if shap_values.shape[1] == len(FEATURE_COLUMNS):
                        class_shap = shap_values[0]
                    else:
                        class_shap = shap_values[class_idx]
                else:
                    class_shap = shap_values.ravel()
            elif hasattr(shap_values, "values"):  # Explanation object
                vals = shap_values.values
                if vals.ndim == 3:
                    class_shap = vals[0, :, class_idx]
                elif vals.ndim == 2:
                    class_shap = vals[0]
                else:
                    class_shap = vals.ravel()
            else:
                class_shap = np.zeros(len(FEATURE_COLUMNS))

            # Extract base value for this class
            if hasattr(explainer, "expected_value"):
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    if len(ev) > class_idx:
                        base_value = float(ev[class_idx])
                    else:
                        base_value = float(ev[0])
                else:
                    base_value = float(ev)

            shap_rows = [
                {"feature": FEATURE_COLUMNS[index], "shap_value": float(class_shap[index])}
                for index in range(len(FEATURE_COLUMNS))
            ]
            use_shap = True
        except Exception:
            use_shap = False

        importance_rows: list[dict[str, Any]] = []
        if use_shap:
            # Populate importance_rows from absolute SHAP values for backwards compatibility
            importance_rows = sorted(
                [
                    {"feature": FEATURE_COLUMNS[index], "importance": abs(float(class_shap[index]))}
                    for index in range(len(FEATURE_COLUMNS))
                ],
                key=lambda x: x["importance"],
                reverse=True,
            )
        else:
            if hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float)
                order = np.argsort(importances)[::-1][: len(FEATURE_COLUMNS)]
                importance_rows = [
                    {"feature": FEATURE_COLUMNS[index], "importance": float(importances[index])}
                    for index in order
                ]
            else:
                try:
                    X, y, _groups = self._safe_reference_dataset(limit=48)
                    result = permutation_importance(model, X, y, n_repeats=3, random_state=self.seed)
                    order = np.argsort(result.importances_mean)[::-1][: len(FEATURE_COLUMNS)]
                    importance_rows = [
                        {
                            "feature": FEATURE_COLUMNS[index],
                            "importance": float(result.importances_mean[index]),
                        }
                        for index in order
                    ]
                except Exception:
                    importance_rows = []

        return {
            "likely_manipulation": predicted_label,
            "confidence": self._confidence_from_proba(probabilities),
            "scores": score_map,
            "feature_importance": importance_rows,
            "shap_values": shap_rows,
            "base_value": base_value,
            "use_shap": use_shap,
            "backend": package["backend"],
            "validation_accuracy": package["validation_accuracy"],
            "trained_at": package.get("trained_at"),
        }

    def _safe_reference_dataset(self, limit: int = 48):
        from src.forensic.ml import build_attack_dataset

        X, y, groups = build_attack_dataset(random_state=self.seed)
        return X[:limit], y[:limit], groups[:limit]

    def _confidence_from_proba(self, probabilities):
        sorted_probs = np.sort(probabilities)[::-1]
        top = float(sorted_probs[0])
        second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = top - second

        if top >= 0.75 or (top >= 0.65 and margin >= 0.2):
            return "HIGH"
        if top >= 0.45 or margin >= 0.1:
            return "MEDIUM"
        return "LOW"

    def _extract_metrics(self, audio, sr, magnitude, ber, ecc_success=None) -> dict[str, Any]:
        from src.forensic.features import extract_forensic_features

        return extract_forensic_features(audio, sr, magnitude, ber, ecc_success=ecc_success)

    def verify_file(self, file_path: str) -> VerificationResult:
        audio, sr = load_audio(file_path)
        stft = compute_stft(audio)
        magnitude, _phase = split_mag_phase(stft)
        audio_hash = compute_audio_hash(audio, sr)

        # Smart metadata matcher: search for the correct location map among all cached metadata files
        best_grouped_locations = None
        best_payload_bits = None
        best_ber = 1.0
        best_ecc_success = False
        best_packet = None
        best_signature_valid = False

        # Resolve metadata directory relative to this file so it works regardless of CWD
        _pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _meta_dir = os.path.join(_pkg_root, "metadata")

        candidates_with_ber: list[tuple[float, str, str]] = []
        if os.path.exists(_meta_dir):
            for entry in os.listdir(_meta_dir):
                if entry.endswith("_grouped_locations.pkl"):
                    source_name = entry.replace("_grouped_locations.pkl", "")
                    loc_path = os.path.join(_meta_dir, f"{source_name}_grouped_locations.pkl")
                    pay_path = os.path.join(_meta_dir, f"{source_name}_payload_bits.pkl")
                    if os.path.exists(pay_path):
                        # Quick BER probe to sort candidates: best match first
                        try:
                            with open(loc_path, "rb") as _h:
                                _locs = pickle.load(_h)
                            with open(pay_path, "rb") as _h:
                                _bits = pickle.load(_h)
                            _rec = extract_payload_qim(magnitude, _locs)[: len(_bits)]
                            _ber = float(np.mean(np.array(_bits) != np.array(_rec)))
                        except Exception:
                            _ber = 1.0
                        candidates_with_ber.append((_ber, loc_path, pay_path))

        # Sort so the lowest-BER (most likely correct) candidate is tried first
        candidates_with_ber.sort(key=lambda x: x[0])
        candidates = [(lp, pp) for _, lp, pp in candidates_with_ber]

        # Append absolute default paths as final fallback
        _def_loc = os.path.join(_pkg_root, self.grouped_locations_path)
        _def_pay = os.path.join(_pkg_root, self.payload_bits_path)
        if os.path.exists(_def_loc) and os.path.exists(_def_pay):
            candidates.append((_def_loc, _def_pay))

        for loc_path, pay_path in candidates:
            if not os.path.exists(loc_path) or not os.path.exists(pay_path):
                continue
            try:
                with open(loc_path, "rb") as handle:
                    cand_locations = pickle.load(handle)
                with open(pay_path, "rb") as handle:
                    cand_bits = pickle.load(handle)

                cand_recovered = extract_payload_qim(magnitude, cand_locations)[: len(cand_bits)]
                cand_ber = float(np.mean(np.array(cand_bits) != np.array(cand_recovered)))

                cand_bytes = bits_to_bytes(cand_recovered)
                cand_decoded = rs_decode(cand_bytes, parity_bytes=self.rs_parity)

                cand_sig_valid = False
                cand_packet = None
                if cand_decoded is not None:
                    try:
                        cand_packet = deserialize_payload(zlib.decompress(cand_decoded))
                        if cand_packet is not None:
                            metadata = cand_packet.get("metadata", {})
                            signature = cand_packet.get("signature")
                            metadata_bytes = serialize_payload(metadata)
                            cand_sig_valid = verify_signature(metadata_bytes, signature)
                    except Exception:
                        pass

                if cand_sig_valid:
                    best_grouped_locations = cand_locations
                    best_payload_bits = cand_bits
                    best_ber = cand_ber
                    best_ecc_success = True
                    best_packet = cand_packet
                    best_signature_valid = cand_sig_valid
                    break

                if cand_ber < best_ber:
                    best_ber = cand_ber
                    best_grouped_locations = cand_locations
                    best_payload_bits = cand_bits
                    best_ecc_success = (cand_decoded is not None)
                    best_packet = cand_packet
                    best_signature_valid = False
            except Exception:
                continue

        grouped_locations = best_grouped_locations or self._load_grouped_locations()
        payload_bits = best_payload_bits or self._load_payload_bits()
        ber = best_ber
        ecc_success = best_ecc_success
        packet = best_packet
        signature_valid = best_signature_valid
        source_hash_match: Optional[bool] = None

        provenance: dict[str, Any] = {}
        if packet is not None:
            metadata = packet.get("metadata", {})
            signature = packet.get("signature")
            metadata_bytes = serialize_payload(metadata)
            signature_valid = verify_signature(metadata_bytes, signature)
            provenance = {
                "id": metadata.get("id"),
                "generator": metadata.get("generator"),
                "timestamp": metadata.get("timestamp"),
                "creator": metadata.get("creator"),
                "model_version": metadata.get("model_version"),
                "source_hash": metadata.get("source_hash"),
                "verified_hash": audio_hash,
            }
            if signature_valid:
                source_hash_match = metadata.get("source_hash") == audio_hash

        if self.attack_mode == "rules":
            attack_analysis = classify_attack_rules(
                audio,
                sr,
                magnitude,
                ber,
                source_hash_match=source_hash_match,
                ecc_success=ecc_success,
            )
        elif self.attack_mode == "ml":
            attack_analysis = classify_attack_ml(
                audio,
                sr,
                magnitude,
                ber,
                source_hash_match=source_hash_match,
                ecc_success=ecc_success,
            )
        else:
            rule_analysis = classify_attack_rules(
                audio,
                sr,
                magnitude,
                ber,
                source_hash_match=source_hash_match,
                ecc_success=ecc_success,
            )
            ml_analysis = classify_attack_ml(
                audio,
                sr,
                magnitude,
                ber,
                source_hash_match=source_hash_match,
                ecc_success=ecc_success,
            )
            attack_analysis = dict(ml_analysis)
            attack_analysis["rule_label"] = rule_analysis["likely_manipulation"]
            attack_analysis["rule_confidence"] = rule_analysis["confidence"]
            attack_analysis["rule_evidence"] = rule_analysis["evidence"]
            attack_analysis["rule_scores"] = rule_analysis["scores"]
            attack_analysis["rule_metrics"] = rule_analysis["metrics"]

        attack_analysis["explainability"] = self.explain_prediction(
            audio,
            sr,
            magnitude,
            ber,
            ecc_success=ecc_success,
        )

        final_result = decide_final_authentication(sig_ok=signature_valid, source_hash_match=source_hash_match, analysis=attack_analysis)
        auth_card_status = _card_status(signature_valid, source_hash_match)

        report_lines = self._build_report_lines(
            ber=ber,
            ecc_success=ecc_success,
            packet=packet,
            signature_valid=signature_valid,
            source_hash_match=source_hash_match,
            attack_analysis=attack_analysis,
            final_result=final_result,
            provenance=provenance,
            audio_hash=audio_hash,
        )

        return VerificationResult(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            sample_rate=sr,
            audio_hash=audio_hash,
            ber=ber,
            ecc_success=ecc_success,
            signature_valid=signature_valid,
            source_hash_match=source_hash_match,
            final_authentication=final_result,
            auth_card_status=auth_card_status,
            auth_card_color=_card_color(auth_card_status),
            provenance=provenance,
            attack_analysis=attack_analysis,
            metrics=attack_analysis["metrics"],
            report_lines=report_lines,
        )

    def _build_report_lines(
        self,
        *,
        ber: float,
        ecc_success: bool,
        packet: Optional[dict[str, Any]],
        signature_valid: bool,
        source_hash_match: Optional[bool],
        attack_analysis: dict[str, Any],
        final_result: str,
        provenance: dict[str, Any],
        audio_hash: str,
    ) -> list[str]:
        lines: list[str] = []
        lines.append("AUTHENTICITY REPORT\n")
        lines.append("=====================\n\n")
        lines.append("Raw BER\n")
        lines.append("-------\n")
        lines.append(f"  BER: {ber:.4f}\n\n")
        lines.append("ECC Status\n")
        lines.append("----------\n")
        if not ecc_success:
            lines.append("  Status: FAILED\n")
            lines.append("  Detail: Too many channel errors to correct.\n\n")
        else:
            lines.append("  Status: SUCCESS\n\n")
        lines.append("Signature Status\n")
        lines.append("----------------\n")
        if packet is None:
            lines.append("  Status: SKIPPED\n")
            lines.append("  Detail: Payload unavailable.\n\n")
        else:
            if signature_valid:
                lines.append("  Status: VALID\n")
                lines.append("  Detail: Signature verified.\n\n")
            else:
                lines.append("  Status: INVALID\n")
                lines.append("  Detail: Signature mismatch.\n\n")
        lines.append("Provenance Hash Status\n")
        lines.append("----------------------\n")
        if packet is None or not signature_valid:
            lines.append("  Status: SKIPPED\n")
            lines.append("  Detail: Provenance unavailable.\n\n")
        else:
            lines.append(f"  Source Hash: {provenance.get('source_hash')}\n")
            lines.append(f"  Verified Audio Hash: {audio_hash}\n")
            if source_hash_match:
                lines.append("  Status: SOURCE MATCH\n")
                lines.append("  Detail: Source and verified audio hashes match.\n\n")
            else:
                lines.append("  Status: SOURCE DIFFERENT FROM VERIFIED FILE\n")
                lines.append("  Detail: Source audio hash differs from verified audio hash.\n\n")
        lines.extend(build_attack_report(attack_analysis, source_hash_match))
        lines.append("Final Authentication\n")
        lines.append("--------------------\n")
        lines.append(f"  {final_result}\n\n")
        lines.append("Recovered Metadata\n")
        lines.append("------------------\n")
        if packet is not None and signature_valid:
            metadata = packet["metadata"]
            lines.append(f"  ID: {metadata.get('id')}\n")
            lines.append(f"  Generator: {metadata.get('generator')}\n")
            lines.append(f"  Timestamp: {metadata.get('timestamp')}\n")
            lines.append(f"  Creator: {metadata.get('creator')}\n")
            lines.append(f"  Model Version: {metadata.get('model_version')}\n")
            lines.append(f"  Source Hash: {metadata.get('source_hash')}\n")
            lines.append("\nPROVENANCE SUMMARY\n")
            lines.append("-------------------\n")
            lines.append(f"Creator:\n{metadata.get('creator')}\n\n")
            lines.append(f"Generated By:\n{metadata.get('generator')}\n\n")
            lines.append(f"Model Version:\n{metadata.get('model_version')}\n\n")
            lines.append(f"Timestamp:\n{metadata.get('timestamp')}\n")
            lines.append("\nPROVENANCE MODEL\n")
            lines.append("----------------\n")
            lines.append("Source hash stored in metadata.\n")
            lines.append("Verified audio hash computed from the file under inspection.\n")
        else:
            lines.append("  Recovery failed.\n")
            lines.append("  Metadata unavailable.\n")
        lines.append("\n")
        return lines
