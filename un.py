# import pickle
# import pandas as pd

# # Load the trained model from the pickle file
# loaded_model = pickle.load(open('model.pkl', 'rb'))

# # Take input values from the user

# # {'pH': 6.867301321799131, 'Hardness': 174.1849763847294, 'Solids': 24112.153470524572,
# #                     'Chloramines': 5.529942235981782, 'Sulfate': 297.6555748707137, 'Conductivity': 484.12293887759057,
# #                     'Organic Carbon': 11.48268699578233, 'Trihalomethanes': 65.30441362142098, 'Turbidity': 3.8235950398688825}


# input_data = [6.867301321799131, 174.1849763847294, 24112.153470524572, 5.529942235981782,
#               297.6555748707137, 484.12293887759057, 11.48268699578233, 65.30441362142098, 3.8235950398688825]

# # Convert the input data to a pandas DataFrame
# input_df = pd.DataFrame([input_data])
# print(input_df)
# # Make a prediction using the loaded model
# prediction = loaded_model.predict(input_df)
# print(prediction)
# print("Input dataframe columns:", input_df.columns)
# # print("Feature dataframe columns:", feature_df.columns)

# # If the prediction is "Not Portable," compare the input values with the features used for training the model
# if prediction[0] == 0:
#     # Create a dictionary or a pandas DataFrame containing the feature values used for training the model
#     feature_data = {'ph': [7.0, 7.5], 'Hardness': [150, 250], 'Solids': [250, 350],
#                     'Chloramines': [2.0, 3.0], 'Sulfate': [200, 300], 'Conductivity': [400, 600],
#                     'Organic Carbon': [8, 12], 'Trihalomethanes': [40, 60], 'Turbidity': [2.5, 4.0]}

#     feature_df = pd.DataFrame(feature_data)

#     # Compare the input values with the feature values used for training the model and store the mismatched values
#     mismatched_data = {}
#     c = 0
#     for column in feature_df.columns:
#         print(column)
#         if input_df[c] < feature_df[column][0] or input_df[c] > feature_df[column][1]:
#             mismatched_data[column] = input_df[c]
#         c += 1

#     # Display the mismatched values to the user
#     print("The following input values do not match the trained model:")
#     print(mismatched_data)


import pickle
import pandas as pd

# Load the trained model from the pickle file
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Take input values from the user
input_data = [6.867301321799131, 174.1849763847294, 24112.153470524572, 5.529942235981782,
              297.6555748707137, 484.12293887759057, 11.48268699578233, 65.30441362142098, 3.8235950398688825]

# Convert the input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])
# print(input_df)

# Make a prediction using the loaded model
prediction = loaded_model.predict(input_df)
print(prediction)
print("Input dataframe columns:", input_df.columns)

# If the prediction is "Not Portable," compare the input values with the features used for training the model
if prediction[0] == 0:
    # Create a dictionary or a pandas DataFrame containing the feature values used for training the model
    feature_data = {'ph': [7.0, 7.5], 'Hardness': [150, 250], 'Solids': [250, 350],
                    'Chloramines': [2.0, 3.0], 'Sulfate': [200, 300], 'Conductivity': [400, 600],
                    'Organic Carbon': [8, 12], 'Trihalomethanes': [40, 60], 'Turbidity': [2.5, 4.0]}

    feature_df = pd.DataFrame(feature_data)

    # Compare the input values with the feature values used for training the model and store the mismatched values
    mismatched_data = {}
    c = 0
    for column in feature_df.columns:
        print(input_df.iloc[0][0])
        if input_df.iloc[0][c] < feature_df[column][0] or input_df.iloc[0][c] > feature_df[column][1]:
            mismatched_data[column] = input_df.iloc[0][c]
        c += 1

    # Display the mismatched values to the user
    if len(mismatched_data) > 0:
        print("The following input values do not match the trained model:")
        print(mismatched_data)
    else:
        print("All input values match the trained model.")
else:
    print("The water is portable.")
