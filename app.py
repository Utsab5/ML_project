import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
from sklearn.preprocessing import MinMaxScaler

# Create flask app
app = Flask(__name__)




model = pickle.load(open("RFR_model2.pkl", "rb"))
scaler = MinMaxScaler()

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():

    # print(x for x in request.form.values())
    # float_features = [float(x) for x in request.form.values()]
    # features = [np.array(float_features)]
    # print(float_features)
    # print(features)
    # prediction = model.predict(features)
    # return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))


    Medu = request.form['Medu']
    Fedu = request.form['Fedu']
    travelTime = request.form['travelTime']
    studyTime = request.form['studyTime']
    failures = request.form['failures']
    famrel = request.form['famrel']
    freetime = request.form['freetime']
    goout = request.form['goout']
    health = request.form['health']
    absences = request.form['absences']
    creativity = request.form['creativity']
    school = request.form['school']
    gender = request.form['gender']
    school_sup = request.form['school_sup']
    family_sup = request.form['family_sup']
    address = request.form['address']
    paid = request.form['paid']
    Activities = request.form['Activities']
    higher = request.form['higher']
    internet = request.form['internet']
    
    form_data = {
    'Medu': request.form['Medu'],
    'Fedu': request.form['Fedu'],
    'travelTime': request.form['travelTime'],
    'studyTime': request.form['studyTime'],
    'failures': request.form['failures'],
    'famrel': request.form['famrel'],
    'freetime': request.form['freetime'],
    'goout': request.form['goout'],
    'health': request.form['health'],
    'absences': request.form['absences'],
    'creativity': request.form['creativity'],
    'school': request.form['school'],
    'gender': request.form['gender'],
    'school_sup': request.form['school_sup'],
    'family_sup': request.form['family_sup'],
    'address': request.form['address'],
    'paid': request.form['paid'],
    'Activities': request.form['Activities'],
    'higher': request.form['higher'],
    'internet': request.form['internet']
    }
    scaler.fit(list(form_data.values()))
    normalized_data = scaler.transform(list(form_data.values()))
    print(normalized_data)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)