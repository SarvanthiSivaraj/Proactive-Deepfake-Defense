def build_attack_report(analysis, source_hash_match):
    metrics = analysis["metrics"]
    lines = []

    lines.append("ATTACK ANALYSIS\n")
    lines.append("---------------\n")
    lines.append(f"Likely Manipulation: {analysis['likely_manipulation']}\n")
    lines.append(f"Confidence: {analysis['confidence']}\n")

    if analysis.get("backend"):
        lines.append(f"Model Backend: {analysis['backend']}\n")

    if analysis.get("validation_accuracy") is not None:
        lines.append(f"Validation Accuracy: {analysis['validation_accuracy']:.3f}\n")

    if analysis.get("rule_label"):
        lines.append(f"Rule Label: {analysis['rule_label']}\n")

    if analysis.get("rule_confidence"):
        lines.append(f"Rule Support: {analysis['rule_confidence']}\n")

    lines.append("Evidence:\n")
    lines.append(f"  BER profile: {metrics['ber']:.4f}\n")
    lines.append(f"  Hash status: {'MATCH' if source_hash_match else 'MISMATCH'}\n")
    lines.append(f"  ECC pressure: {metrics['ecc_pressure']:.4f}\n")
    lines.append(
        f"  Spectral difference: centroid={metrics['centroid']:.2f}, hf_ratio={metrics['hf_ratio']:.6f}, flatness={metrics['flatness']:.6f}\n"
    )
    lines.append(f"  Energy drift: rms={metrics['rms']:.6f}, peak={metrics['peak']:.6f}\n")
    lines.append(
        f"  Synchronization shift: edge_ratio={metrics['edge_ratio']:.6f}, column_shift_pressure={metrics['column_shift_pressure']:.6f}\n\n"
    )

    if analysis.get("rule_evidence"):
        lines.append("Rule Evidence:\n")
        for item in analysis["rule_evidence"]:
            lines.append(f"  - {item}\n")
        lines.append("\n")

    return lines


def decide_final_authentication(sig_ok, source_hash_match, analysis):
    if not sig_ok:
        return "NOT AUTHENTIC"

    # Signature is valid — the file is cryptographically authenticated.
    # source_hash_match=True  → exact original file is verified (AUTHENTIC)
    # source_hash_match=False → a watermarked/derivative of the original (AUTHENTIC)
    # source_hash_match=None  → hash check skipped (still AUTHENTIC if sig is valid)
    return "AUTHENTIC"
