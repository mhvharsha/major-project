from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib


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




# Creating model object
model_kn = KNeighborsClassifier(n_neighbors=9, leaf_size=20)

# Training Model
model_kn.fit(X_train, Y_train)

# Making Prediction
pred_kn = model_kn.predict(X_test)

# Calculating Accuracy Score
kn = accuracy_score(Y_test, pred_kn)
print(kn)

# Save the trained model using joblib
joblib.dump(model_kn, 'knn_model.pkl')
