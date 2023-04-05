import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load data
df = pd.read_csv('csv/water_potability.csv')
df.fillna(df.mean(), inplace=True)
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Preprocess data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




# Create SVM model with RBF kernel
model = SVC(kernel='rbf', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Save the model using pickle
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print(f'Accuracy: {accuracy:.2f}')
