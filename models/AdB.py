from functions import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

# Define the AdaBoost model
base_model = DecisionTreeClassifier(max_depth=1)
n_estimators = 5000
learning_rate = 0.001
model = AdaBoostClassifier(estimator=base_model, n_estimators=n_estimators, learning_rate=learning_rate)

data = import_uci()
data = prepare_data(data)

# Split the dataset into training and test sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Using stratified splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the AdaBoost model
model.fit(X_train, y_train)

# Predict probabilities for test set
y_prob = model.predict_proba(X_test)

# Calculate AUC-ROC score
auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')


# Evaluate the AdaBoost model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\nAccuracy = {accuracy:.4f}\nPrecision = {precision:.4f}\nRecall = {recall:.4f}\nF1 Score = {f1:.4f}")
print(f"\nAUC-ROC Score = {auc_roc:.4f}")