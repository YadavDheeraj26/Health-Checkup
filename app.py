from lib2to3.pgen2.grammar import opmap_raw
from flask import Flask,render_template,request
import pickle
import numpy as np
import joblib


app = Flask(__name__)

# loading the models
breast_m = pickle.load(open('mdoel_beast.pkl','rb'))
diabetes_m = pickle.load(open('model_diabetes.pkl','rb'))
heart_m = joblib.load(open('heart_model.pkl','rb'))
kidney_m = pickle.load(open('model_kidney.pkl','rb'))
liver_m = pickle.load(open('model_liver.pkl','rb'))


# breast-cancer
@app.route("/b_cancer")
def b_cancer():
    return render_template('/breast_cancer/cancer.html')

# liver
@app.route("/liver")
def lung_cancer():
    return render_template('/liver/liver.html')

# kidney
@app.route("/kidney")
def kidney():
    return render_template('/kidney/kidney.html')

# heart
@app.route("/heart")
def heart():
    return render_template('/heart/heart.html')

# diabetes
@app.route("/diabetes")
def diabetes():
    return render_template('/diabetes/diabetes.html')

@app.route("/")
def home():
    return render_template('index.html')


# making routes for the model to pedict

@app.route('/predict_heart',methods=['POST'])
def predict_heart():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    output=heart_m.predict(final_input)

    if(int(output)==1):
        prediction = "Sorry you may have chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return render_template("./heart/heart.html",prediction_text=prediction)


@app.route('/predict_cancer',methods=['POST'])
def predict_cancer():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    output=breast_m.predict(final_input)

    if(int(output)==1):
        prediction = "You have the Malignent Tumor"
    else:
        prediction = "You have Benign Tumor"

    return render_template("./breast_cancer/cancer.html",prediction_text=prediction)


@app.route('/predict_diabetes',methods=['POST'])
def predict_diabetes():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    output=diabetes_m.predict(final_input)

    if(int(output)==1):
        prediction = "You are having a great chances of diabetes. Please consult to Doctor"
    else:
        prediction = "Feel Free , You are Safe"

    return render_template("./diabetes/diabetes.html",prediction_text=prediction)


@app.route('/predict_kidney',methods=['POST'])
def predict_kidney():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    output=kidney_m.predict(final_input)

    if(int(output)==1):
        prediction = "You are having a great chances of Kidney Diseases. Please consult to Doctor"
    else:
        prediction = "Feel Free , You are Safe"

    return render_template("./kidney/kidney.html",prediction_text=prediction)

@app.route('/predict_liver',methods=['POST'])
def predict_liver():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    output=liver_m.predict(final_input)

    if(int(output)==1):
        prediction = "You are having a great chances of Liver Diseases. Please consult to Doctor"
    else:
        prediction = "Feel Free , You are Safe"

    return render_template("./liver/liver.html",prediction_text=prediction)


if __name__ == "__main__":

    app.run(debug=True)
