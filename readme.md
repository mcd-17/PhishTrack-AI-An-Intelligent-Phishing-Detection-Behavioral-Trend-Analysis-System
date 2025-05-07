# 🛡️ PhishTrack AI

PhishTrack AI is an intelligent phishing detection system that uses machine learning to analyze URLs, text content, and visual/behavioral patterns to flag potential phishing threats. The system is modular, secure, and designed for backend integration, logging phishing trends for administrative insights.

---

## 📌 Features

- 🔗 **URL Phishing Detection** using ML models.
- ✍️ **Text-based Phishing Analysis** with language and obfuscation scanners.
- 🖼️ **Screenshot Similarity Detection** using CNNs for phishing lookalikes.
- 🔁 **Link Redirection Chain Analyzer** for detecting suspicious redirect patterns.
- 📈 **Behavioral & Temporal Trend Logging** for long-term analysis.
- 🌐 **RESTful API** endpoints for integration.

---

## 📁 Project Structure

| **Directory/File**                          | **Description**                             |
| ------------------------------------------- | ------------------------------------------- |
| **backend/**                                | Root directory for the backend code         |
| ├── **app.py**                              | Flask app entry point                       |
| ├── **utils/**                              | Utility functions and models                |
| │ ├── **url\_model.py**                     | URL phishing model logic                    |
| │ ├── **text\_model.py**                    | Text phishing model logic                   |
| │ ├── **report\_generator.py**              | Report creation logic                       |
| │ ├── **language\_detector.py**             | Detects language of content                 |
| │ ├── **anti\_obfuscation\_model.py**       | Obfuscation detection                       |
| │ ├── **visual\_similarity\_model.py**      | CNN-based screenshot phishing detection     |
| │ ├── **url\_redirect\_chain\_analyzer.py** | Full redirect chain analyzer                |
| │ ├── **link\_redirect\_model.py**          | Lightweight redirect checker                |
| │ └── **temporal\_analysis\_model.py**      | Trend logging module                        |
| ├── **data/**                               | Contains data files                         |
| │ └── **report\_trends.csv**                | Phishing activity logs                      |
| ├── **models/**                             | Contains trained models                     |
| │ └── **phishing\_url\_model.pkl**          | Trained ML model for URL phishing detection |
| └── **requirements.txt**                    | Lists project dependencies                  |



---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/phishtrack-ai.git
cd phishtrack-ai/backend 

### 2. Install Dependencies

pip install -r requirements.txt


### 3. Run the Application

python app.py



🔌 API Endpoints

| Endpoint              | Method | Description                            |
| --------------------- | ------ | -------------------------------------- |
| `/check_url`          | POST   | Analyze a URL for phishing             |
| `/check_text`         | POST   | Analyze text content                   |
| `/detect_obfuscation` | POST   | Detect obfuscated code                 |
| `/detect_language`    | POST   | Identify the language used             |
| `/generate_report`    | POST   | Create a full phishing analysis report |


📊 Trend Logging
Each phishing detection event is timestamped and logged to data/report_trends.csv via the log_phishing_event() function. This enables admin users to visualize and track phishing trends over time


📌 Authors
Darshana M Chigari — Project Lead & Developer



Note-  PhishTrack AI is a machine learning-based phishing detection system that I have developed, designed to analyze URLs, text content, and visual/behavioral patterns to flag potential phishing threats. The project includes several key features such as URL phishing detection, text analysis, visual similarity checks, and link redirection chain analysis. Although the core logic is built and a basic RESTful API has been implemented, the project is currently incomplete due to some technical errors, particularly with model integration and prediction processes. These issues are being actively addressed, but they have delayed the full completion of the system.

While PhishTrack AI is still a work in progress, it has been a significant learning experience. I’ve gained insights into phishing detection, machine learning techniques, and backend integration. Despite encountering challenges along the way, I’m dedicated to resolving these issues and enhancing the project. Once complete, the system will offer a more accurate phishing detection service and provide valuable insights through trend logging and reporting.
