# utils/domain_reputation_model.py

import whois
import requests
from collections import defaultdict
import json
import logging

# Domain reputation sources
BLACKLIST_API_URL = "https://someblacklistapi.com/check"  # Replace with a real API
WHOIS_SERVER = "whois.iana.org"  # You can replace this with other WHOIS servers if needed

# Logging configuration
logging.basicConfig(level=logging.INFO)

class DomainReputationModel:
    def __init__(self):
        self.blacklist = set()
        self.load_blacklist()

    def load_blacklist(self):
        """
        Load the blacklist from a file or external source.
        For now, we'll simulate it with some predefined domains.
        """
        # Here, you could load a list of known bad domains from a local file, a database, or an external service
        self.blacklist = {"badwebsite.com", "phishingdomain.com", "maliciousurl.org"}
        logging.info("Loaded blacklist.")

    def check_blacklist(self, domain):
        """
        Check if the domain is in the blacklist.
        """
        return domain.lower() in self.blacklist

    def check_whois(self, domain):
        """
        Check WHOIS information for the domain to identify suspicious indicators.
        Suspicious domains might have characteristics such as:
        - Recent registration
        - Missing or obscured owner details
        - Non-standard domain TLDs
        """
        try:
            whois_info = whois.whois(domain)
            if whois_info.creation_date:
                # Check if the domain was registered within the past year (example criteria)
                age = (whois_info.creation_date - whois_info.updated_date).days
                if age < 365:
                    logging.warning(f"Domain {domain} has been recently registered.")
                    return True  # Potentially suspicious domain due to recent registration
            return False  # Safe domain
        except Exception as e:
            logging.error(f"Error checking WHOIS for {domain}: {str(e)}")
            return False

    def check_reputation(self, domain):
        """
        Combine blacklist and WHOIS checks to determine domain reputation.
        """
        # Check in the local blacklist first
        if self.check_blacklist(domain):
            logging.info(f"Domain {domain} is blacklisted.")
            return {"reputation": "bad", "reason": "blacklist"}

        # Check WHOIS information
        if self.check_whois(domain):
            logging.info(f"Domain {domain} is suspicious based on WHOIS information.")
            return {"reputation": "suspicious", "reason": "new_registration"}

        # If no issues found, consider the domain safe
        logging.info(f"Domain {domain} appears to be safe.")
        return {"reputation": "good", "reason": "reputable"}

    def get_domain_reputation(self, domain):
        """
        Integrate all checks and provide a final reputation score.
        """
        # You can add more complex checks here if needed (e.g., integrating third-party APIs)
        reputation = self.check_reputation(domain)
        return reputation


# Example Usage:
# Initialize domain reputation model
domain_model = DomainReputationModel()

# Check the reputation of a domain
domain = "maliciouswebsite.com"
reputation = domain_model.get_domain_reputation(domain)
print(f"Domain {domain} reputation: {reputation['reputation']} ({reputation['reason']})")
