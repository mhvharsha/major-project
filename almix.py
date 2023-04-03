
import pickle
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('csv/water_potability.csv')

# Data Cleaning
df.dropna(inplace=True)

# Handling imbalanced data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(
    df.drop('Potability', axis=1), df['Potability'])

# Feature Engineering
skb = SelectKBest(f_classif, k=5)
X_resampled = skb.fit_transform(X_resampled, y_resampled)

# Split data into training and testing sets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the individual models
model1 = make_pipeline(
    SelectKBest(f_classif),
    GradientBoostingClassifier(random_state=42))

model2 = make_pipeline(
    PowerTransformer(),
    PCA(),
    RandomForestClassifier(random_state=42))

model3 = make_pipeline(
    StandardScaler(),
    MLPClassifier(random_state=42))

model4 = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42))

model5 = make_pipeline(
    StandardScaler(),
    SVC(random_state=42))

model6 = make_pipeline(
    StandardScaler(),
    XGBClassifier(random_state=42))

# Define the hyperparameter grids for grid search
param_grid1 = {
    'gradientboostingclassifier__n_estimators': [50, 100, 200],
    'gradientboostingclassifier__learning_rate': [0.01, 0.1, 1],
    'gradientboostingclassifier__max_depth': [3, 5, 7]
}

param_grid2 = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [3, 5, 7]
}

param_grid3 = {
    'mlpclassifier__hidden_layer_sizes': [(10,), (50,), (100,)],
    'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
    'mlpclassifier__solver': ['sgd', 'adam'],
    'mlpclassifier__alpha': [0.0001, 0.001, 0.01]
}

param_grid4 = {
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__C': [0.1, 1, 10],
    'logisticregression__solver': ['liblinear']
}

param_grid5 = {
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}

param_grid6 = {
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1, 1],
    'xgbclassifier__n_estimators': [50, 100, 200]
}

# Create a list of the individual models
models = [('gbc', model1),
          ('rf', model2),
          ('mlp', model3),
          ('logreg', model4),
          ('svm', model5),
          ('xgb', model6)]

# Define the voting classifier with the individual models and their hyperparameter grids
voting_clf = VotingClassifier(estimators=models, voting='hard')

param_grid_voting = {
    'voting': ['hard'],
    'gbc__gradientboostingclassifier__n_estimators': [50, 100, 200],
    'gbc__gradientboostingclassifier__learning_rate': [0.01, 0.1, 1],
    'gbc__gradientboostingclassifier__max_depth': [3, 5, 7],
    'rf__randomforestclassifier__n_estimators': [50, 100, 200],
    'rf__randomforestclassifier__max_depth': [3, 5, 7],
    'mlp__mlpclassifier__hidden_layer_sizes': [(10,), (50,), (100,)],
    'mlp__mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
    'mlp__mlpclassifier__solver': ['sgd', 'adam'],
    'mlp__mlpclassifier__alpha': [0.0001, 0.001, 0.01],
    'logreg__logisticregression__penalty': ['l1', 'l2'],
    'logreg__logisticregression__C': [0.1, 1, 10],
    'logreg__logisticregression__solver': ['liblinear'],
    'svm__svc__kernel': ['linear', 'rbf', 'poly'],
    'svm__svc__C': [0.1, 1, 10],
    'svm__svc__gamma': ['scale', 'auto'],
    'xgb__xgbclassifier__max_depth': [3, 5, 7],
    'xgb__xgbclassifier__learning_rate': [0.01, 0.1, 1],
    'xgb__xgbclassifier__n_estimators': [50, 100, 200]
}


# Define grid search with cross-validation
grid_search = GridSearchCV(
    voting_clf, param_grid_voting, cv=skf, scoring='accuracy')

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best model and print its accuracy on the testing data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of the best model on the testing data is {accuracy:.3f}')

# Save the best model to a file
pickle.dump(best_model, open(
    "best_model.pkl", 'wb'))


# Save the scaler and skb to files
pickle.dump(best_model, open(
    "scaler_model.pkl", 'wb'))
pickle.dump(best_model, open(
    "skb_model.pkl", 'wb'))
