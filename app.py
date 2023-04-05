import cloudpickle as pickle
from flask import Flask, render_template, request, request, jsonify

import pickle
import joblib
import pandas as pd
import numpy as np

from un import misMatchFeatures

app = Flask(__name__)


@app.route("/")
def flask():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    name = ""
    if request.method == "POST":
        name = request.form.get("userName")
        print(f"name: {name}")
    return render_template("submit.html", n=name)


filename = 'combined_model.pkl'
with open(filename, 'rb') as f:
    loaded_model = joblib.load(f)


def check_input(input_dict, range_dict):
    for key, value in range_dict.items():
        if not (value[0] <= float(input_dict[key]) <= value[1]):
            return False
    return True


@app.route('/predict', methods=['POST'])
def predict():

    ph = request.form.get('ph')
    hardness = request.form.get('hardness')
    solids = request.form.get('solids')
    chloramines = request.form.get('chloramines')
    sulfate = request.form.get('sulfate')
    conductivity = request.form.get('conductivity')
    organic_carbon = request.form.get('organic_carbon')
    trihalomethanes = request.form.get('trihalomethanes')
    turbidity = request.form.get('turbidity')

    input_values = [ph, hardness, solids, chloramines, sulfate,
                    conductivity, organic_carbon, trihalomethanes, turbidity]
    data = [input_values]
    input_dict = {'ph': ph, 'Hardness': hardness, 'Solids': solids, 'Chloramines': chloramines,
                  'Sulfate': sulfate, 'Conductivity': conductivity, 'Organic Carbon': organic_carbon,
                  'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity}

    range_dict = {'ph': [6.5, 8.5], 'Hardness': [150, 250], 'Solids': [250, 350],
                  'Chloramines': [2.0, 3.0], 'Sulfate': [200, 300], 'Conductivity': [400, 600],
                  'Organic Carbon': [8, 12], 'Trihalomethanes': [40, 60], 'Turbidity': [2.5, 4.0]}

    # if check_input(input_dict, range_dict) == False:
    #     return render_template("undrinkable.html", n=str(range_dict))

    data_numeric = np.array(data).astype(float)

    prediction = loaded_model.predict(data_numeric)

    prediction_value = str(prediction[0])
    if (prediction_value == 0):
        a = misMatchFeatures(input_values)
        return render_template("submit.html", n=str(a))

    return render_template("submit.html", n=str(prediction_value))


if __name__ == "__main__":
    app.run(debug=True)
