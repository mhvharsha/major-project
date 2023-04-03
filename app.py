from flask import Flask, render_template, request, request, jsonify

import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route("/hi")
def flask():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    name = ""
    if request.method == "POST":
        name = request.form.get("userName")
        print(f"name: {name}")
    return render_template("submit.html", n=name)


filename = 'svm_model.pkl'
# Load the model from the pickle file
loaded_model = pickle.load(open(filename, 'rb'))


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

    prediction = loaded_model.predict(
        [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

    prediction_value = str(prediction[0])

    return prediction_value
    # return render_template("submit.html", n=ph)


if __name__ == "__main__":
    app.run(debug=True)
