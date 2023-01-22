import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# Create flask app
app = Flask(__name__)

def norm(data):
    val = data.values()
    final = list(val)    
    x_array =np.array(final)
    normalized_arr = preprocessing.normalize([x_array])
    return normalized_arr


# app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///mlProject.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
# db=SQLAlchemy(app)


# class SPFModel(db.Model):
#     sno=db.Column(db.Integer,primary_key=True)
#     title=db.Column(db.String(200),nullable=False)

#     def __repr__(self) -> str:
#         return f"{self.sno} - {self.title}"


model = pickle.load(open("RFR_model2.pkl", "rb"))
scaler = MinMaxScaler()

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():

    # print(x for x in request.form.values())
    # float_features = [float(x) for x in request.form.values()]
    
    # print(float_features)
    # print(features)
    


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

    if school == "GP":
        school_GP = 1
        school_MS = 0
    else:
        school_GP = 0
        school_MS = 1


    if gender == "M":
        gender_M = 1
        gender_F = 0
    else:
        gender_F = 1
        gender_M = 0

    if address == "R":
        address_R=1
        address_U = 0
    else:
        address_R=0
        address_U =1

    if school_sup == "yes":
        schoolsup_no=0
        schoolsup_yes=1
    else:
        schoolsup_no=1
        schoolsup_yes=0



    if family_sup == "yes":
        famsup_no=0
        famsup_yes=1
    else:
        famsup_no=1
        famsup_yes=0

    if paid == "yes":
        paid_no=0
        paid_yes=1
    else:
        paid_no=1
        paid_yes=0


    if Activities == "yes":
        activities_no=0
        activities_yes=1
    else:
        activities_no=1
        activities_yes=0


    if higher == "yes":
        higher_no=0
        higher_yes=1
    else:
        higher_no=1
        higher_yes=0

    if internet == "yes":
        internet_no=0
        internet_yes=1
    else:
        internet_no=0
        internet_yes=1


    
    form_data = {
    'Medu': (request.form['Medu']),
    'Fedu': (request.form['Fedu']),
    'traveltime': request.form['travelTime'],
    'studytime': request.form['studyTime'],
    'failures': request.form['failures'],
    'famrel': request.form['famrel'],
    'freetime': request.form['freetime'],
    'goout': request.form['goout'],
    'health': request.form['health'],
    'absences': request.form['absences'],
    'creativity': request.form['creativity'],
    'school_GP': school_GP,
    'school_MS': school_MS,
    'sex_F': gender_F,
    'sex_M': gender_M,
    'address_R': address_R,
    'address_U': address_U,
    'schoolsup_no': schoolsup_no,
    'schoolsup_yes': schoolsup_yes,    
    'famsup_no': famsup_no,
    'famsup_yes': famsup_yes,    
    'paid_no': paid_no,
    'paid_yes': paid_yes,
    'activities_no': activities_no,
    'activities_yes': activities_yes,
    'higher_no':higher_no,
    'higher_yes':higher_yes,
    'internet_no': internet_no,
    'internet_yes': internet_yes
    }
    
    # print(norm(form_data))
    # features = [np.array(form_data)]

    form_data_array = np.array(list(form_data.values()))
    form_data_array = form_data_array.reshape(1, -1)
    output = model.predict(form_data_array)

    # prediction = model.predict(form_data)
    return render_template("index.html", prediction_text = "The Grade {}".format(output))

    # return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
