from flask import Flask, jsonify, request
from pro125 import getPrediction

app = Flask(__name__)

@app.route("/predict-digit", methods=["POST"])
def predict_data():
  # image = cv2.imdecode(np.fromstring(request.files.get("digit").read(), np.uint8), cv2.IMREAD_UNCHANGED)
  image = request.files.get("digit")
  prediction = getPrediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)