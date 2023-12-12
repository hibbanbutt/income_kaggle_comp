import csv

import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as stats

from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load train and test data
train = pd.read_csv('data/train_final.csv', na_values='?')
test = pd.read_csv('data/test_final.csv', na_values='?')

# Separate label from rest of train data
X = train.drop('income>50K', axis=1)
y = train['income>50K']

# Separate ID from rest of test data
X_test = test.drop('ID', axis=1)
id = test['ID']

# Impute missing numerical values using the mean
numerical_imputer = SimpleImputer(strategy='mean')
X_numeric = numerical_imputer.fit_transform(X.select_dtypes(include=['number']))
X_test_numeric = numerical_imputer.fit_transform(X_test.select_dtypes(include=['number']))

# Impute missing categorical values using the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical = categorical_imputer.fit_transform(X.select_dtypes(exclude=['number']))
X_test_categorical = categorical_imputer.fit_transform(X_test.select_dtypes(exclude=['number']))

# Encode categorical values using one hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical)
X_test_categorical_encoded = encoder.transform(X_test_categorical)

# Reconstruct the entire datasets, now encoded and complete
X_combined = hstack((X_numeric, X_categorical_encoded))
X_test_combined = hstack((X_test_numeric, X_test_categorical_encoded))

######HYPERPARAM
# Define a distribution of hyperparameters to sample from
param_dist = {
    'n_estimators': np.arange(100, 1000, 50),
    'max_depth': np.arange(3, 20),
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': np.linspace(0.5, 1.0, 10),
    'colsample_bytree': np.linspace(0.5, 1.0, 10),
    'gamma': np.linspace(0, 0.5, 10),
}

xgb_model = xgb.XGBClassifier()


# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist,
                                   scoring='accuracy', cv=5, n_iter=5000, n_jobs=-1, verbose=10)

# Fit the random search to the data
random_search.fit(X_combined, y)

# Get the best parameters and the best estimator
best_params = random_search.best_params_
best_score = random_search.best_score_
best_xgb_model = random_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_xgb_model.predict(X_test_combined)

# Calculate accuracy on the test set
print("Best Hyperparameters:", best_params)
print("Best Score:", best_score)

######HYPERPARAM


# Train
# xgb_classifier = xgb.XGBClassifier(n_estimators=90, random_state=42)
# xgb_classifier.fit(X_combined, y)


# Predict on test data
result = {'ID': list(id), 'Prediction': list(y_pred)}

# Save result to file in kaggle comp required format
with open("fourth_attempt.csv", "w") as output:
    writer = csv.writer(output)
    headers = list(result.keys())
    limit = len(result[headers[0]])
    writer.writerow(result.keys())

    for i in range(limit):
        writer.writerow([result[x][i] for x in headers])

