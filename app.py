import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("ndex.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    print(features)
    prediction = model.predict(features)
    return render_template("ndex.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)