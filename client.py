import requests
import time
import os

# Config
SERVER_URL = "http://127.0.0.1:8000/data"  # ML server endpoint
IMAGE_DIR = "esp_images/"                    # folder with images
SEND_INTERVAL = 5  # seconds between sends

# List all images in folder
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg") or f.endswith(".png")]

for img_name in images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    with open(img_path, "rb") as f:
        files = {"image": f}
        try:
            response = requests.post(SERVER_URL, files=files)
            data = response.json()
            print(f"Sent {img_name}, received: {data}")
        except Exception as e:
            print(f"Error sending {img_name}: {e}")
    
    time.sleep(SEND_INTERVAL)
