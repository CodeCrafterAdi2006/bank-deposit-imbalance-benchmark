import os
import zipfile
import urllib.request
import pandas as pd

# URLs and directory paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
ZIP_PATH = os.path.join(DATA_DIR, "bank.zip")
TARGET_CSV = os.path.join(DATA_DIR, "bank-full.csv")

# UCI Bank Marketing Dataset download URL
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Download parent archive if not present
    if not os.path.exists(ZIP_PATH) and not os.path.exists(TARGET_CSV):
        print(f"[*] Downloading Bank Marketing dataset from UCI...")
        urllib.request.urlretrieve(UCI_URL, ZIP_PATH)
        print(f"[+] Download complete: {ZIP_PATH}")
    
    # 2. Extract bank-full.csv (handling nested zip if present)
    if not os.path.exists(TARGET_CSV):
        print(f"[*] Extracting archives to {DATA_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as outer_zip:
            outer_zip.extractall(DATA_DIR)
            
        # The UCI zip typically contains an inner 'bank.zip'
        inner_zip_path = os.path.join(DATA_DIR, "bank.zip")
        if os.path.exists(inner_zip_path) and inner_zip_path != ZIP_PATH:
            with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
                inner_zip.extractall(DATA_DIR)
        elif os.path.exists(os.path.join(DATA_DIR, "bank-full.csv")):
            pass
        else:
            # Check if inner bank.zip exists in the extracted directory
            for item in os.listdir(DATA_DIR):
                if item.endswith(".zip") and os.path.join(DATA_DIR, item) != ZIP_PATH:
                    with zipfile.ZipFile(os.path.join(DATA_DIR, item), 'r') as z:
                        z.extractall(DATA_DIR)
                        
    # 3. Validate extracted dataset
    if os.path.exists(TARGET_CSV):
        df = pd.read_csv(TARGET_CSV, sep=';')
        print(f"\n[+] Success! Dataset loaded: {TARGET_CSV}")
        print(f"    - Total Rows: {df.shape[0]}")
        print(f"    - Total Features: {df.shape[1]}")
        print(f"    - Class Balance (y):\n{df['y'].value_counts(normalize=True).apply(lambda x: f'{x*100:.2f}%')}")
    else:
        raise FileNotFoundError(f"Could not locate bank-full.csv in {DATA_DIR}")

if __name__ == "__main__":
    download_and_extract()
