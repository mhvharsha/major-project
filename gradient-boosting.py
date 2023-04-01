
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


model = GradientBoostingClassifier()
n_estimators = [50, 100, 150, 200, 250]
max_depth = [3, 5, 7, 9]
min_samples_split = [2, 3, 4]
learning_rate = [0.01, 0.1, 0.5, 1]


grid = dict(n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, learning_rate=learning_rate)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
grid_search_gb = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                              scoring='accuracy', error_score=0)
grid_search_gb.fit(X_train, Y_train)


print(
    f"Best: {grid_search_gb.best_score_:.3f} using {grid_search_gb.best_params_}")
means = grid_search_gb.cv_results_['mean_test_score']
stds = grid_search_gb.cv_results_['std_test_score']
params = grid_search_gb.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")


def classification_report_with_accuracy_score(Y_test, y_pred2):
    # print classification report
    print(classification_report(Y_test, y_pred2))
    return accuracy_score(Y_test, y_pred2)  # return accuracy score


nested_score = cross_val_score(grid_search_gb, X=X_train, y=Y_train, cv=cv,
                               scoring=make_scorer(classification_report_with_accuracy_score))
print(nested_score)

gb_y_predicted = grid_search_gb.predict(X_test)

gb_y_predicted

gb_grid_score = accuracy_score(Y_test, gb_y_predicted)

print(gb_grid_score)

grid_search_gb.best_params_

confusion_matrix(Y_test, gb_y_predicted)


X_GB = gb.predict([[5.735724, 158.318741, 25363.016594, 7.728601,
                  377.543291, 568.304671, 13.626624, 75.952337, 4.732954]])

print(X_GB)
