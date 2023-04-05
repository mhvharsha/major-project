
def misMatchFeatures(input_data):

    # If the prediction is "Not Portable," compare the input values with the features used for training the model

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
            print(feature_data[column])
            mismatched_data[column] = [
                feature_df[column][0], feature_df[column][1]]
        c += 1

    # Display the mismatched values to the user
    if len(mismatched_data) > 0:
        print("The following input values do not match the trained model:")
        print(mismatched_data)
    else:
        print("All input values match the trained model.")


# misMatchFeatures(input_data)
