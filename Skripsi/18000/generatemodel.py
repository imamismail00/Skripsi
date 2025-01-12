import numpy as np
import pandas as pd
import pickle
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Path to your datasets
filepath1 = "TRAIN-DATA.csv"
filepath2 = "18000.csv"

# Load datasets
train = pd.read_csv(filepath1)
test = pd.read_csv(filepath2)

# Preprocessing
train = train.dropna()
test = test.dropna()

# Encode labels
le = LabelEncoder()

# Separate features and target
x_train = train.drop(columns=['label'])
y_train = train['label'].values

x_test = test.drop(columns=['label'])
y_test = test['label'].values

x_train = x_train.apply(le.fit_transform)
y_train = le.fit_transform(y_train)

x_test = x_test.apply(le.fit_transform)
y_test = le.fit_transform(y_test)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize the AdaBoost Classifier
base_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5)  # Base model for boosting
adaboost_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=20,  # Number of weak learners
    learning_rate=1.0  # Learning rate
)

# Record the start time for training
start_time = datetime.datetime.now()

# Train the AdaBoost model
adaboost_model.fit(x_train_scaled, y_train)

# Record the end time for training
end_time = datetime.datetime.now()

# Calculate training time
training_time = end_time - start_time

# Make predictions
y_pred = adaboost_model.predict(x_test_scaled)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = metrics.recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_score = metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)

print("-------------------------------------")
print("AdaBoost Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Training Time:", training_time)
print("-------------------------------------")

# Save the model
filename = "ensemble_boosting2.sav"
pickle.dump(adaboost_model, open(filename, 'wb'))

print("AdaBoost model saved as", filename)

