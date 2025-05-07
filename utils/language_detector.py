# utils/language_detector.py

from langdetect import detect, detect_langs
# Removed LangDetectException as it is not available in langdetect

class LanguageDetector:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'ru': 'Russian',
            'zh-cn': 'Chinese (Simplified)',
            'ar': 'Arabic',
            'hi': 'Hindi',
        }

    def detect_language(self, text):
        """Detects the most probable language of the input text."""
        try:
            lang_code = detect(text)
            return self.supported_languages.get(lang_code, f"Unknown ({lang_code})")
        except Exception:
            return "Could not detect language"

    def detect_language_probs(self, text):
        """Returns a list of probable languages with confidence scores."""
        try:
            return detect_langs(text)
        except LangDetectException:
            return ["Could not detect"]

# For standalone testing
if __name__ == "__main__":
    detector = LanguageDetector()
    test_text = "Hola, ¿cómo estás?"
    print("Detected Language:", detector.detect_language(test_text))
    print("Language Probabilities:", detector.detect_language_probs(test_text))
