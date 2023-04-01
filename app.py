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


filename = 'model.pkl'
# Load the model from the pickle file
loaded_model = pickle.load(open(filename, 'rb'))
# Define a POST method that receives an input JSON and returns the prediction


@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON input
    # data = request.get_json(force=True)
    # # Convert the input to a dataframe
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)
    # # Make a prediction using the loaded model
    # prediction = loaded_model.predict(data_df)
    # # Return the prediction as a JSON
    # output = {'prediction': int(prediction[0])}
    # print("output")
    # return jsonify(output)
    # data = request.json
    ph = request.form.get('ph')
    hardness = request.form.get('hardness')
    solids = request.form.get('solids')
    chloramines = request.form.get('chloramines')
    sulfate = request.form.get('sulfate')
    conductivity = request.form.get('conductivity')
    organic_carbon = request.form.get('organic_carbon')
    trihalomethanes = request.form.get('trihalomethanes')
    turbidity = request.form.get('turbidity')
    # input_features = [float(x) for x in request.form.values()]
    # feature_names = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    #                  "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability"]
    # df = pd.DataFrame(feature_names, columns=feture_names)

    # df = scaler.transform(df)
#     print("jio")
# # Now you can use the loaded model to make predictions on this data
    prediction = loaded_model.predict(
        [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

    prediction_value = str(prediction[0])

# Return the prediction as a JSON object
    return prediction_value
    return render_template("submit.html", n=ph)


if __name__ == "__main__":
    app.run(debug=True)
