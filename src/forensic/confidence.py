def estimate_confidence(scores):
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = best_score - sorted_scores[1] if len(sorted_scores) > 1 else best_score

    if best_score >= 5 or (best_score >= 4 and margin >= 2):
        return "HIGH"

    if best_score >= 2 or margin >= 1:
        return "MEDIUM"

    return "LOW"
