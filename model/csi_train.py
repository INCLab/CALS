import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from dataloader import DataLoader

dataPath = '../data/pe'

# # Load Person Exist dataset
# pe_df, npe_df = DataLoader().loadPEdata(dataPath)
#
# csi_df = pd.concat([pe_df, npe_df], ignore_index=True)
#
# print('< PE data size > \n {}'.format(len(pe_df)))
# print('< NPE data size > \n {}'.format(len(npe_df)))
#
#
# # Divide feature and label
# csi_data, csi_label = csi_df.iloc[:, 2:-1], csi_df.iloc[:, -1]
#

# Load Person Exist dataset
pe_df, npe_df = DataLoader().loadWindowPeData(dataPath)

csi_df = pd.concat([pe_df, npe_df], ignore_index=True)

print('< PE data size > \n {}'.format(len(pe_df)))
print('< NPE data size > \n {}'.format(len(npe_df)))


# Divide feature and label
csi_data, csi_label = csi_df.iloc[:, :-1], csi_df.iloc[:, -1]


# Divide Train, Test dataset
X_train, X_test, y_train, y_test = train_test_split(csi_data, csi_label, test_size=0.2, random_state=42)

# Sampling
# SAMPLE_NUM = 35000
# X_train, y_train = X_train[:SAMPLE_NUM], y_train[:SAMPLE_NUM]
# X_test, y_test = X_test[:int(SAMPLE_NUM * 0.2)], y_test[:int(SAMPLE_NUM * 0.2)]


print("< X_train shape >")
print(X_train.shape)

print("< y_train shape >")
print(y_train.shape)

# RF
params = { 'svm_clf__n_estimators' : [100],
           'svm_clf__max_depth' : [12],
           'svm_clf__min_samples_leaf' : [8],
           'svm_clf__min_samples_split' : [20]
            }

kernel_rf_clf = Pipeline([

    # ("scaler", StandardScaler()),
    ("svm_clf", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=8,
                                       min_samples_split=20, random_state=0, n_jobs=-1))
])


# grid_cv = GridSearchCV(kernel_rf_clf, param_grid = params, cv = 3, n_jobs = -1)
# grid_cv.fit(X_train, y_train)

kernel_rf_clf.fit(X_train, y_train)

pred = kernel_rf_clf.predict(X_test)
score = kernel_rf_clf.score(X_test, y_test)

print(score)
print(classification_report(y_test, pred))
