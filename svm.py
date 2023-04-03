from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC, LinearSVC

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv(r'csv\water_potability.csv')

df.fillna(df.mean(), inplace=True)
df['ph'] = df['ph'].fillna(df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(
    df['Trihalomethanes'].mean())

X = df.drop('Potability', axis=1)
Y = df['Potability']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)


model_svm = SVC(kernel='rbf', random_state=42)
model_svm.fit(X_train, Y_train)
SVC(random_state=42)
# Making Prediction
pred_svm = model_svm.predict(X_test)
# Calculating Accuracy Score
sv = accuracy_score(Y_test, pred_svm)
pickle.dump(model_svm, open("svm_model.pkl", 'wb'))


print(sv)
# joblib.dump(model_svm, 'svm_model.pkl')
