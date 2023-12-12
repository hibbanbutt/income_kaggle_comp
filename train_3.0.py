import csv
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as stats
import optuna

from optuna.samplers import TPESampler
from numpy import NaN
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# Load train and test data
train = pd.read_csv('data/train_final.csv', na_values='?').replace(NaN, 'NaN')
test = pd.read_csv('data/test_final.csv', na_values='?').replace(NaN, 'NaN')

# Separate label from rest of train data
X = train.drop('income>50K', axis=1)
y = train['income>50K']

# Separate ID from rest of test data
X_kaggle = test.drop('ID', axis=1)
id = test['ID']

# Impute missing numerical values using the mean
# numerical_imputer = SimpleImputer(strategy='mean')
# X_numeric = X.select_dtypes(include=['number'])
# X_numeric[:] = numerical_imputer.fit_transform(X_numeric)
#
# X_test_numeric =  X_test.select_dtypes(include=['number'])
# X_test_numeric[:] = numerical_imputer.fit_transform(X_test_numeric)
#
# # Impute missing categorical values using the most frequent value
# categorical_imputer = SimpleImputer(strategy='most_frequent')
# X_categorical = X.select_dtypes(exclude=['number'])
# X_categorical[:] = categorical_imputer.fit_transform(X_categorical)
#
# X_test_categorical = X_test.select_dtypes(exclude=['number'])
# X_test_categorical[:] = categorical_imputer.fit_transform(X_test_categorical)
#
# print(X_categorical)
# # Encode categorical values using one hot encoding
# # encoder = OneHotEncoder(handle_unknown='ignore')
# # X_categorical_encoded = encoder.fit_transform(X_categorical)
# # X_test_categorical_encoded = encoder.transform(X_test_categorical)
#
# # Reconstruct the entire datasets, now encoded and complete
# X_combined = pd.concat([X_numeric, X_categorical], axis=1, join='inner')
# print(X_combined)
# X_test_combined = pd.concat([X_test_numeric, X_test_categorical], axis=1, join='inner')

# def objective(trial):
#     cv_folder = StratifiedKFold(n_splits=5, random_state=1,shuffle=True)
#
#     model = CatBoostClassifier(
#         iterations=trial.suggest_int("iterations", 100, 1000),
#         learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
#         depth=trial.suggest_int("depth", 4, 10),
#         l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
#         bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
#         random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
#         bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
#         od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
#         od_wait=trial.suggest_int("od_wait", 10, 50),
#         cat_features = X.select_dtypes(include=['object']).columns.tolist(),
#         scale_pos_weight = len(y[y==0]) / len(y[y==1])
#     )
#
#     return cross_val_score(model, X, y, cv=cv_folder, scoring="f1_weighted").mean()
#
#
# sampler = TPESampler(seed=1)
# study = optuna.create_study(study_name="catboost", direction="maximize", sampler=sampler)
# study.optimize(objective, n_trials=100)
#
# print("Number of finished trials: ", len(study.trials))
# print("Best trial:")
# trial = study.best_trial
# print("  Value: ", trial.value)
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {} = {},".format(key, value))

#Train
model_final = CatBoostClassifier(    iterations = 995,
    learning_rate = 0.03460391162425298,
    depth = 10,
    l2_leaf_reg = 1.0376932803995385,
    bootstrap_type = "Bayesian",
    random_strength = 2.3403625257862575,
    bagging_temperature = 0.6891131355880675,
    od_type = "Iter",
    od_wait = 50,
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    )

model_final.fit(X, y)

y_pred_final = model_final.predict(X_kaggle)

# Predict on test data
result = {'ID': list(id), 'Prediction': list(y_pred_final)}

# Save result to file in kaggle comp required format
with open("catboost_optimized2.csv", "w") as output:
    writer = csv.writer(output)
    headers = list(result.keys())
    limit = len(result[headers[0]])
    writer.writerow(result.keys())

    for i in range(limit):
        writer.writerow([result[x][i] for x in headers])

