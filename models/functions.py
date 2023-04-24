import pandas as pd
import numpy as np

def import_uci():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, header=None, names=column_names)
    return data

def prepare_data(data):
    # Replace "?" values with NaN values
    data.replace("?", np.nan, inplace=True)
    # Impute missing values with column means
    data = data.astype(float)
    means = data.mean()
    data.fillna(means, inplace=True)
    return data

def print_results(scores, accuracy, precision, recall, f1, auc_roc):
    print("\nResults:")
    # Print the cross-validation scores
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", np.mean(scores))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("AUC-ROC:", auc_roc)
