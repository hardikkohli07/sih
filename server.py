from flask import Flask, request, jsonify
from predict import predict  # make sure predict() returns (class_idx, label)
import requests

app = Flask(__name__)

# --------------------------
# Config
# --------------------------
RECEIVER_URL = "http://127.0.0.1:6000/receieve"  # Receiver endpoint
DEFAULT_LAT = 28.7041   # Replace with actual GPS if available
DEFAULT_LON = 77.1025

# --------------------------
# Endpoint
# --------------------------
@app.route('/data', methods=['POST'])
def classify_image():
    # Get uploaded image
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    img_file = request.files['image']
    
    # Check if file is empty
    if img_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Run prediction
    try:
        pred_class, pred_label = predict(img_file)
        
        # Handle prediction failure
        if pred_class is None or pred_label is None:
            return jsonify({"error": "Prediction failed"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Build payload
    payload = {
        "class": int(pred_class),
        "label": pred_label,
        "lat": DEFAULT_LAT,
        "lon": DEFAULT_LON
    }

    # Forward to receiver (make this optional)
    try:
        r = requests.post(RECEIVER_URL, json=payload, timeout=5)
        if r.status_code == 200:
            print("✅ Forwarded result to receiver successfully")
        else:
            print(f"⚠ Forwarding failed, status: {r.status_code}")
    except Exception as e:
        print(f"⚠ Error forwarding result: {e}")
        # Don't fail the main request if forwarding fails

    # Return JSON to client
    return jsonify(payload)

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
