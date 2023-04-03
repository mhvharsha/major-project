import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv(r'csv\water_potability.csv')

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Split the data into training and testing sets
X = df.drop('Potability', axis=1)
Y = df['Potability']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=101, shuffle=True)

# Define the SVM model
svm_model = SVC()

# Define a narrower range of hyperparameters for the grid search
C = [0.1, 1]
gamma = [0.1, 'scale']
kernel = ['linear', 'rbf']

grid = dict(C=C, gamma=gamma, kernel=kernel)

# Use a simpler cross-validation strategy with fewer splits and iterations
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Perform the grid search
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=grid, n_jobs=-1, cv=cv,
                               scoring='accuracy', error_score=0)
grid_search_svm.fit(X_train, Y_train)

# Print the best parameters and mean test score
print(
    f"Best: {grid_search_svm.best_score_:.3f} using {grid_search_svm.best_params_}")

# Print the mean test score and standard deviation for each set of hyperparameters
means = grid_search_svm.cv_results_['mean_test_score']
stds = grid_search_svm.cv_results_['std_test_score']
params = grid_search_svm.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")

# Define a function to print the classification report and return the accuracy score


def classification_report_with_accuracy_score(Y_test, y_pred2):
    print(classification_report(Y_test, y_pred2))
    return accuracy_score(Y_test, y_pred2)


# Use the simpler cross-validation strategy with fewer splits and iterations for nested cross-validation
nested_score = cross_val_score(grid_search_svm, X=X_train, y=Y_train, cv=cv,
                               scoring=make_scorer(classification_report_with_accuracy_score))
print(nested_score)

# Evaluate the model on the testing set
svm_y_predicted = grid_search_svm.predict(X_test)
svm_grid_score = accuracy_score(Y_test, svm_y_predicted)
print(svm_grid_score)

# Print the confusion matrix
confusion_matrix(Y_test, svm_y_predicted)

# Predict on a new data point
X_svm = svm_model.predict([[5.735724, 158.318741, 25363.016594, 7.728601,
                            377.543291, 568.304671, 13.626624, 75.952337, 4.732954]])
print(X_svm)
