import requests
from urllib.parse import urlparse
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# List of known phishing domains (example: add more in real-world use)
PHISHING_DOMAINS = ['phishing.com', 'malicious.com', 'fakebank.com']

# Function to get the redirect chain
def get_redirect_chain(url):
    try:
        # Send a GET request to the URL and allow redirects
        response = requests.get(url, allow_redirects=True)
        
        # List of all redirects
        redirect_chain = response.history + [response]
        
        # Extract and log the URLs in the chain
        chain_urls = [r.url for r in redirect_chain]
        logging.info(f"Redirect chain for {url}: {chain_urls}")
        
        return chain_urls
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return []

# Function to analyze the redirect chain for phishing risk
def analyze_redirect_chain(url):
    redirect_chain = get_redirect_chain(url)
    
    # Check for suspicious redirect behavior
    if len(redirect_chain) > 5:
        logging.warning(f"Suspicious: More than 5 redirects for {url}.")
        return True  # Potential phishing
    
    # Check if the final URL or any part of the chain points to a known phishing domain
    for redirect_url in redirect_chain:
        parsed_url = urlparse(redirect_url)
        domain = parsed_url.netloc
        
        if any(phishing_domain in domain for phishing_domain in PHISHING_DOMAINS):
            logging.warning(f"Phishing detected in the redirect chain: {redirect_url}")
            return True  # Potential phishing

    logging.info(f"No phishing detected for {url}.")
    return False

# Example of testing a URL (can be triggered via backend later)
if __name__ == "__main__":
    test_url = "http://example.com"  # Replace with a test URL
    result = analyze_redirect_chain(test_url)
    if result:
        print(f"The URL {test_url} has a suspicious redirect chain!")
    else:
        print(f"The URL {test_url} appears safe.")
