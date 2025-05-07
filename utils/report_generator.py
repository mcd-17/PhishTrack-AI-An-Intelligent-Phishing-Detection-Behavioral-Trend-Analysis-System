import datetime
import uuid

class ReportGenerator:
    def __init__(self):
        pass

    def generate_report(self, url_result=None, text_result=None):
        report_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()

        report = {
            "report_id": report_id,
            "generated_at": timestamp,
            "url_analysis": url_result if url_result else "Not provided",
            "text_analysis": text_result if text_result else "Not provided",
            "summary": self._create_summary(url_result, text_result)
        }

        return report

    def _create_summary(self, url_result, text_result):
        if url_result and url_result.get("result") == "phishing":
            return "⚠️ Suspicious URL detected."
        elif text_result and text_result.get("result") == "phishing":
            return "⚠️ Suspicious text content detected."
        elif url_result or text_result:
            return "✅ Content appears safe."
        else:
            return "❓ No data available for analysis."
