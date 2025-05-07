import joblib 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import make_pipeline 
urls = [
    "http://example.com",                         # 0 (legit)
    "https://secure-login.bank.com",             # 1 (phishing-like)
    "http://phishing.com",                       # 0 (legit but misleading domain)
    "https://mybank.com",                        # 1 (assumed phishing for this demo)
    "https://e-zpass.com-txcr.world/",           # 1 (phishing)
    "https://e-zpass.com-iapo.world/",           # 1 (phishing)
    "https://driveezmd.com-ygfd.win/",           # 1 (phishing)
    "https://e-zpass.com-txck.xyz/"              # 1 (phishing)
]
labels = [0, 1, 0, 1] 
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier()) 
model.fit(urls, labels) 
joblib.dump(model, 'models/phishing_url_model.pkl') 
