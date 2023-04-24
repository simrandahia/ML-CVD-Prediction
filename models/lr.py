from functions import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

data = import_uci()
data = prepare_data(data)

# Split the dataset into training and test sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Using stratified splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the logistic regression model using 10-fold cross-validation
logreg = LogisticRegression(max_iter=10000000)
model = logreg

# Using stratified cross-validation
cv = StratifiedKFold(n_splits=10)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

# Fit the SVM model on the training set
model.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy, precision, recall, and F1 score of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate AUC-ROC
y_score = model.decision_function(X_test)
y_prob = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')


print_results(scores, accuracy, precision, recall, f1, auc_roc)
