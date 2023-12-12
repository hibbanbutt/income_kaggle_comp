import csv
import pandas as pd
import xgboost as xgb
import scipy.stats as stats

from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV


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

# Train
xgb_classifier = xgb.XGBClassifier(n_estimators=90, random_state=42)
xgb_classifier.fit(X_combined, y)

# Predict on test data
xgb_predictions = xgb_classifier.predict(X_test_combined)
result = {'ID': list(id), 'Prediction': list(xgb_predictions)}

# Save result to file in kaggle comp required format
with open("second_attempt.csv", "w") as output:
    writer = csv.writer(output)
    headers = list(result.keys())
    limit = len(result[headers[0]])
    writer.writerow(result.keys())

    for i in range(limit):
        writer.writerow([result[x][i] for x in headers])

