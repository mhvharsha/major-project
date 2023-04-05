import pickle
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

# Load data
df = pd.read_csv('csv/water_potability.csv')
df.fillna(df.median(), inplace=True)

# Split data into training and testing sets
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define the individual models
model1 = KNeighborsClassifier()
model2 = SVC(probability=True)
model3 = XGBClassifier()


# Define the hyperparameter grids for grid search
param_grid1 = {'n_neighbors': [3, 5, 7]}
param_grid2 = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
param_grid3 = {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]}

# Perform grid search to find the best hyperparameters for each model
grid1 = GridSearchCV(model1, param_grid1, cv=5, n_jobs=-1)
grid2 = GridSearchCV(model2, param_grid2, cv=5, n_jobs=-1)
grid3 = GridSearchCV(model3, param_grid3, cv=5, n_jobs=-1)

grid1.fit(X_train, y_train)
grid2.fit(X_train, y_train)
grid3.fit(X_train, y_train)


model1 = grid1.best_estimator_
model2 = grid2.best_estimator_
model3 = grid3.best_estimator_

# Define the voting classifier
ensemble = VotingClassifier(
    estimators=[('knn', model1), ('svc', model2), ('xgb', model3)], voting='soft')

# Train the voting classifier on the data
ensemble.fit(X_train, y_train)

# Make predictions using the voting classifier
y_pred = ensemble.predict(X_test)

# Print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save the model
joblib.dump(ensemble, 'combined_model.pkl')
