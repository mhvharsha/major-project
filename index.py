from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from flask import Flask, request, jsonify
import pickle

# Load the dataset
df = pd.read_csv(r'csv\water_potability.csv')

# Fill in missing values with mean
df.fillna(df.mean(), inplace=True)

# Partitioning
X = df.drop('Potability', axis=1)
Y = df['Potability']

# Split the dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=101, shuffle=True)

# Train the model
dt = DecisionTreeClassifier(
    criterion='gini', min_samples_split=10, splitter='best')
dt.fit(X_train, Y_train)

# Store the model in a pickle file
filename = 'model.pkl'
pickle.dump(dt, open(filename, 'wb'))





