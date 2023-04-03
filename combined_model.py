import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.impute import KNNImputer  # import the KNNImputer

# Load data
df = pd.read_csv(r'csv\water_potability.csv')

# Perform KNN imputation on the data
imputer = KNNImputer(n_neighbors=10)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split data into training and testing sets
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101, shuffle=True)

# Define the individual models
model1 = KNeighborsClassifier(n_neighbors=5)
model2 = SVC(kernel='linear', probability=True)


# Define the voting classifier
ensemble = VotingClassifier(
    estimators=[('knn', model1), ('svc', model2)], voting='hard')

# Train the voting classifier on the data
ensemble.fit(X_train, y_train)

# Make predictions using the voting classifier
y_pred = ensemble.predict(X_test)

# Print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save the model
# joblib.dump(ensemble, 'voting_classifier_model.joblib')
with open('voting_classifier_model.pkl', 'wb') as file:
    pickle.dump(ensemble, file)
