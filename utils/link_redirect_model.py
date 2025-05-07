import requests
from urllib.parse import urlparse
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# List of known phishing domains (this can be expanded)
PHISHING_DOMAINS = ['phishing.com', 'malicious.com', 'fakebank.com']

# Function to check a URL's redirection and flag suspicious links
def check_redirect(url):
    try:
        # Send a GET request to the URL and allow redirects
        response = requests.get(url, allow_redirects=True, timeout=10)
        
        # Get the final URL after redirection
        final_url = response.url
        logging.info(f"Final URL after redirection: {final_url}")
        
        # Check if the final URL or any of the intermediate redirects is suspicious
        redirect_chain = [r.url for r in response.history] + [final_url]
        
        # Log each URL in the redirection chain
        logging.info(f"Redirection chain for {url}: {redirect_chain}")
        
        # Check if the final URL or any part of the chain points to a known phishing domain
        for redirect_url in redirect_chain:
            parsed_url = urlparse(redirect_url)
            domain = parsed_url.netloc
            
            if any(phishing_domain in domain for phishing_domain in PHISHING_DOMAINS):
                logging.warning(f"Phishing detected in the redirect chain: {redirect_url}")
                return True  # Potential phishing
        
        logging.info(f"No phishing detected for {url}.")
        return False  # Safe link

    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking URL {url}: {e}")
        return False  # Treat as a safe URL in case of errors (for fallback handling)

# Example of testing a URL (can be triggered via backend later)
if __name__ == "__main__":
    test_url = "http://example.com"  # Replace with a test URL
    result = check_redirect(test_url)
    if result:
        print(f"The URL {test_url} has suspicious redirects!")
    else:
        print(f"The URL {test_url} appears safe.")
