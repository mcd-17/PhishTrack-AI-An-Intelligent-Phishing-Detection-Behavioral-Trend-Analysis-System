from flask import Flask, request, jsonify
import sys
import os

# Ensure the 'utils' directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import your custom utility classes
from url_model import PhishingURLDetector
from text_model import PhishingTextScanner
from report_generator import ReportGenerator
from anti_obfuscation_model import ObfuscationDetector
from language_detector import LanguageDetector
from temporal_analysis_model import log_phishing_event  # ✅ NEW IMPORT

# Initialize Flask app
app = Flask(__name__)

# Initialize utility classes
url_detector = PhishingURLDetector()
text_scanner = PhishingTextScanner()
report_generator = ReportGenerator()
obfuscation_detector = ObfuscationDetector()
language_detector = LanguageDetector()

# Routes
@app.route('/check_url', methods=['POST'])
def check_url():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    result = url_detector.check_url(url)

    # ✅ Log phishing event if detected
    if result.get('label') == 'phishing':
        log_phishing_event(source='url')

    return jsonify(result)


@app.route('/check_text', methods=['POST'])
def check_text():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    result = text_scanner.check_text(text)

    # ✅ Log phishing event if detected
    if result.get('label') == 'phishing':
        log_phishing_event(source='text')

    return jsonify(result)


@app.route('/detect_obfuscation', methods=['POST'])
def detect_obfuscation():
    data = request.get_json()
    content = data.get('content')

    if not content:
        return jsonify({'error': 'Content is required'}), 400

    obf_result = obfuscation_detector.detect_obfuscation(content)
    return jsonify({'obfuscation_detected': obf_result})


@app.route('/detect_language', methods=['POST'])
def detect_language():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    language = language_detector.detect_language(text)
    return jsonify({'language': language})


@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.get_json()
    url = data.get('url')
    text = data.get('text')

    url_result = url_detector.check_url(url) if url else None
    text_result = text_scanner.check_text(text) if text else None
    obfuscation_result = obfuscation_detector.detect_obfuscation(text or url or "")
    language_result = language_detector.detect_language(text or "") if text else None

    # ✅ Log phishing event if detected by either model
    if url_result and url_result.get('label') == 'phishing':
        log_phishing_event(source='url')
    if text_result and text_result.get('label') == 'phishing':
        log_phishing_event(source='text')

    report = report_generator.generate_report(
        url_result,
        text_result,
        extra_info={
            'obfuscation_detected': obfuscation_result,
            'language_detected': language_result
        }
    )
    return jsonify(report)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
