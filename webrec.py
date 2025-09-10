from flask import Flask, request, jsonify

app = Flask(__name__)

# This endpoint simulates your website or database receiving data
@app.route("/receieve", methods=["POST"])
def receive_data():
    data = request.json
    print(f"Received data: {data}")
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
