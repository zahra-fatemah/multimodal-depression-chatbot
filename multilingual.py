from langdetect import detect
from deep_translator import GoogleTranslator

def translate_to_english(text):
    try:
        lang = detect(text)

        if lang != "en":
            translated = GoogleTranslator(source=lang, target="en").translate(text)
            return translated, lang

        return text, "en"

    except Exception:
        return text, "unknown"
