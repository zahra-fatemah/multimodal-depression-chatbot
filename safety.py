def crisis_detect(text):
    crisis_words = [
        "kill myself",
        "suicide",
        "end my life",
        "i want to die",
        "self harm"
    ]
    return any(word in text.lower() for word in crisis_words)
