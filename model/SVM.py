import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from dataloader import DataLoader

dataPath = '../data/pe'

# Load Person Exist dataset
pe_df, npe_df = DataLoader().loadPEdata(dataPath)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))


# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]


# Sliding Window ====================================================================

# # Load Person Exist dataset
# pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)

# csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

# print('< PE data size > \n {}'.format(len(pe_df)))
# print('< NPE data size > \n {}'.format(len(npe_df)))


# # Divide feature and label
# csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]

# =========================================================================================

# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42)

# Sampling
# SAMPLE_NUM = 35000
# X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
# X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]

print('Total data size: {}'.format(len(csi_data)))
print("< X_train shape >")
print(X_train.shape)

print("< y_train shape >")
print(y_train.shape)

# SVM
params = { 'C' : [10],
           'coef0' : [1],
           'degree' : [4],
           'gamma' : [0.1],
           'kernel' : ['poly']
            }

kernel_svm_clf = Pipeline([
    ("svm_clf", SVC(C=10, coef0=1, degree=4, gamma=0.1, kernel='poly'))
])


# grid_cv = GridSearchCV(kernel_svm_clf, param_grid = params, cv = 3, n_jobs = -1)
# grid_cv.fit(X_train, y_train)

kernel_svm_clf.fit(X_train, y_train)

pred = kernel_svm_clf.predict(X_test)
score = kernel_svm_clf.score(X_test, y_test)
print(score)

print(classification_report(y_test, pred))


