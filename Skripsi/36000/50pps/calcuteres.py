import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing

def getresult(input, output, indices, indicess, name):
    filepath1 = input
    filepath2 = "36000.csv"

    testsdn = pd.read_csv(filepath1)
    test = pd.read_csv(filepath2)

    testsdn = testsdn.dropna()
    test = test.dropna()

    test = test.drop_duplicates()
    temp = len(test)
    test = test[test.index.isin(indices)]
    testsdn = testsdn[testsdn.index.isin(indicess)]

    le = preprocessing.LabelEncoder()

    y_sdn = testsdn['label'].values
    y_test = test['label'].values

    y_test = le.fit_transform(y_test)

    print("----------------------------------------")
    print(classification_report(y_sdn, y_test, zero_division=0))  # Adding zero_division parameter
    print("Accuracy:", accuracy_score(y_sdn, y_test) * 100)
    print("Precision:", precision_score(y_sdn, y_test, average='macro', zero_division=0) * 100)  # Adding zero_division parameter
    print("Recall:", recall_score(y_sdn, y_test, average='macro', zero_division=0) * 100)  # Adding zero_division parameter
    print("F1 Score:", f1_score(y_sdn, y_test, average='macro', zero_division=0) * 100)  # Adding zero_division parameter

    print(confusion_matrix(y_sdn, y_test))

    temp = float(36000 - float(len(indices)))
    temp = float(temp / 36000)
    print("Packet Loss: " + str(temp * 100))

np.set_printoptions(suppress=True)

id = np.load('indexreal.npy')
id = id.astype(int)
print(id.shape)
print(id.dtype)

ids = np.load('indexsims.npy')
ids = ids.astype(int)
print(ids.shape)
print(ids.dtype)

getresult('ensemble_boosting.csv', 'ensemble_boosting', id, ids, 'dt')

