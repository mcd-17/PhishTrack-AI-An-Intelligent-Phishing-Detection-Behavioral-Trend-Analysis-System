import pandas as pd # type: ignore
import os
from datetime import datetime  # Importing standard library module
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Define the path to the data file
DATA_PATH = os.path.abspath(os.path.join("data", "report_trends.csv"))

def log_phishing_event(source: str = "url"):
    """
    Logs a phishing detection event with timestamp and source (e.g., 'url', 'text').
    """
    try:
        # Get the current date
        now = datetime.now().strftime("%Y-%m-%d")
        # Create a DataFrame with the event data
        df = pd.DataFrame([[now, source]], columns=["date", "source"])

        # Ensure the directory exists before any file operation
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

        # Append to the file if it exists, otherwise create a new file
        if os.path.exists(DATA_PATH):
            df.to_csv(DATA_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(DATA_PATH, index=False)
    except Exception as e:
        print(f"Error logging phishing event: {e}. Source: {source}")

def analyze_phishing_trends(show_plot=True):
    """
    Reads the CSV and visualizes phishing trends over time.
    Returns summary DataFrame for optional report use.
    """
    try:
        # Check if the data file exists
        if not os.path.exists(DATA_PATH):
            print("No data available.")
            return None

        # Read the data and preprocess it
        try:
            df = pd.read_csv(DATA_PATH)
        except pd.errors.EmptyDataError:
            print("The data file is empty.")
            return None
        except pd.errors.ParserError:
            print("The data file is corrupted.")
            return None

        df['date'] = pd.to_datetime(df['date'])
        summary = df.groupby(['date', 'source']).size().unstack(fill_value=0)

        # Check if the summary is empty
        if summary.empty:
            print("No data to plot.")
            return None

        # Plot the trends if requested
        if show_plot:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=summary)
            plt.title("Phishing Detection Trends Over Time")
            plt.xlabel("Date")
            plt.ylabel("Detections")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return summary
    except Exception as e:
        print(f"Error analyzing phishing trends: {e}")
        return None
