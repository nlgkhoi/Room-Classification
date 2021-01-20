from flask import Flask, render_template, request, jsonify
import predict_image

app = Flask(__name__)

@app.route('/')
def home():
    return "Room classifier API"

@app.route("/predict")
def get():
    try:
        urls = request.json
        predictions, times = predict_image.batch_predict(urls)
        return jsonify(label=list(zip(urls, predictions, times))), 200
    except:
        return jsonify(label="cannot predict"), 404

if __name__ == '__main__':
    app.run(debug=True)