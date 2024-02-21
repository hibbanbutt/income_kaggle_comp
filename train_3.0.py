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

