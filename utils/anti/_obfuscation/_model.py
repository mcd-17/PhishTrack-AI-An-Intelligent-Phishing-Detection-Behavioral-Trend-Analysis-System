import re
from unidecode import unidecode
import ftfy

class ObfuscationDetector:
    def __init__(self):
        # List of known suspicious patterns or homoglyphs
        self.suspicious_patterns = [
            r'[\u202E\u202D\u202A\u202B]',   # Unicode control characters (e.g., RTL override)
            r'(?:[0-9]+\.\s*){2,}',         # Obfuscated numbers or bullet formats
            r'[^\x00-\x7F]',                # Non-ASCII characters
        ]
        self.homoglyph_map = {
            'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',  # Cyrillic
            'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K',  # Greek
        }

    def normalize_text(self, text):
        """Fix Unicode issues and replace homoglyphs with standard characters."""
        fixed_text = ftfy.fix_text(text)
        ascii_text = unidecode(fixed_text)

        for homoglyph, replacement in self.homoglyph_map.items():
            ascii_text = ascii_text.replace(homoglyph, replacement)

        return ascii_text

    def detect_obfuscation(self, text):
        """Return True if obfuscation is detected, else False."""
        normalized_text = self.normalize_text(text)

        for pattern in self.suspicious_patterns:
            if re.search(pattern, normalized_text):
                return True
        return False

# For standalone testing
if __name__ == "__main__":
    detector = ObfuscationDetector()
    sample = "рayраl.com\u202E"
    print("Obfuscation Detected:", detector.detect_obfuscation(sample))
