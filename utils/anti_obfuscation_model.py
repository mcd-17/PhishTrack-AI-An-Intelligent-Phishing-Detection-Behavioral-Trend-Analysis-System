import re
import unicodedata

class ObfuscationDetector:
    def __init__(self):
        # Patterns that might indicate obfuscation
        self.suspicious_patterns = [
            r'%[0-9a-fA-F]{2}',    # URL-encoded characters
            r'\\x[0-9a-fA-F]{2}',  # Hex-encoded
            r'[\u202E\u200F\u200E]',  # Unicode RTL/LTR characters
            r'<script.*?>',        # Script tags
            r'eval\s*\(',          # Eval usage
            r'document\.write',    # JS injection
            r'<iframe.*?>',        # Hidden iframes
        ]

    def detect_obfuscation(self, content: str) -> bool:
        if not content:
            return False

        # Normalize Unicode characters
        normalized_content = unicodedata.normalize('NFKC', content)

        for pattern in self.suspicious_patterns:
            if re.search(pattern, normalized_content, re.IGNORECASE):
                return True

        return False
