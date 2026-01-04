def multimodal_fusion(text_risk=None, audio_risk=None, video_risk=None):
    weights = {"text": 0.5, "audio": 0.3, "video": 0.2}
    score = 0
    total_weight = 0

    if text_risk is not None:
        score += weights["text"] * text_risk
        total_weight += weights["text"]

    if audio_risk is not None:
        score += weights["audio"] * audio_risk
        total_weight += weights["audio"]

    if video_risk is not None:
        score += weights["video"] * video_risk
        total_weight += weights["video"]

    final_score = score / total_weight

    if final_score < 0.8:
        return "LOW"
    elif final_score < 1.5:
        return "MEDIUM"
    else:
        return "HIGH"
