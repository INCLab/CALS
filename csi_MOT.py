from database.tracking_db import tracking_db
import numpy as np
import pandas as pd

# =========  CSI labeling  =========
label = {}
label_list = []
threshold = 0.9  # error threshold
db = tracking_db()
csi_datas = db.get_csi()
mot_datas = dict(db.get_mot())


# Datalist에서 Value와 가장 유사한 값을 찾음
def matching_values(datalist, value):
    # Threshold 범위 내의 값만 Numpy Array에 담음
    array = datalist[np.where(abs(datalist - value) <= threshold)]

    # 해당하는 값이 하나 이상이면
    if array.size > 0:
        # 그 중 가장 작은 값(minimum error)의 Index를 리턴
        minvalue = np.argmin(abs(array - value))

        return array[minvalue]
    else:
        # 하나도 없다면 -1
        return -1


# MOT Timestamp List (nparray)
mot_times = np.asarray(list(map((lambda x: x), mot_datas.keys())))

for csi_data in csi_datas:
    # 두 값을 비교해 현재 CSI Time과 가장 유사한 MOT Time을 구함
    compare = matching_values(mot_times, csi_data[0])

    # -1이면 매칭 실패
    if compare == -1:
        label[csi_data[0]] = -1
        label_list.append(-1)
    # 그 외에는 해당하는 MOT Timestamp가 Return 되므로 그 값을 Label에 담음
    else:
        target_num = mot_datas[compare]  # total target number in the frame
        exist_label = int()  # (target_num == 0, 0) ,(target_num > 0, 1)

        if target_num == 0:
            exist_label = 0
        elif target_num > 0:
            exist_label = 1

        label[csi_data[0]] = exist_label
        label_list.append(exist_label)

        # 이미 뽑은건 제거
        mot_times = np.delete(mot_times, np.argwhere(mot_times == compare))

print(label_list)
csi_df = db.get_csi_df()

csi_df['label'] = label_list
print(csi_df)

# ========== 학습 모델 ========== #
#
# import pandas as pd
# # 모델 생성되면 저장하는 애(파이참 껐다 켰을 때 다시 학습안하고 기존에 시켜논 모델 불러와서 쓸 수 있음)
# import pickle
# import warnings
#
# from mlxtend.evaluate import confusion_matrix
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, KFold, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
#
# warnings.filterwarnings('ignore')
#
# col = list(range(0, 91))  # 채널 수 0부터 91까지(0(타임스탬프) 1-90(CSI 채널)
# df = pd.read_csv('outputs.csv', encoding='utf-8', names=col)
#
# print(label)
# print(len(df[0]))
#
# df.drop([df.columns[0]], axis=1, inplace=True)  # 시간 정보 필요없으니까 날려
# df['label'] = label
#
# X = df.drop(['label'], 1)
# y = df['label']
#
# # train 75%, valid 25%
# x_train, x_valid, y_train, y_valid = train_test_split(X, y)
#
# # PCA 차원 축소(써보고 테스트 데이터셋으로 성능 높아지는 지 선택적으로 하기)
# pca = PCA(n_components=6)  # 학습 column 90개에서 6개로 column 축소
# principalComponents = pca.fit_transform(x_train)
# nx_train = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'])
#
# principalComponents = pca.fit_transform(x_valid)
# nx_valid = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'])
#
# print("PCA 후 분산 : ", round(sum(pca.explained_variance_ratio_), 3))
#
# # RF 파라미터
# parameter = {
#     'n_estimators': [100, 200],
#     'max_depth': [6, 8, 10, 12],
#     'min_samples_leaf': [3, 5, 7, 10],
#     'min_samples_split': [2, 3, 5, 10]}
#
# name = 'wifiSensing.asv'  # 저장할 모델 이름
#
# # ====== storage model(처음 저장할 때) ====== #
# kfold = KFold(10, shuffle=True)
#
#
# print("======================RF======================")
# # RF
# rf = RandomForestClassifier(random_state=0)
#
#
# # 최적의 모델 생성
# rf_grid = GridSearchCV(rf, param_grid=parameter, scoring="accuracy", n_jobs=-1, cv=kfold)
# rf_grid.fit(x_train, y_train)
#
# pickle.dump(rf_grid, open(name, 'wb'))
#
# # 검증
# prediction = rf_grid.predict(x_valid)
# total_param = rf_grid.cv_results_['params']
# total_score = rf_grid.cv_results_['mean_test_score']
#
# print('best parameter: ', rf_grid.best_params_)
# print('best score: %.2f' % rf_grid.best_score_)
#
# rf_best = rf_grid.best_estimator_
#
# # 검증
# prediction = rf_best.predict(x_valid)
#
# print('score : {:.4f}'.format(accuracy_score(y_valid, prediction)))
# print(confusion_matrix(y_valid, prediction))
# print(classification_report(y_valid, prediction))
#
#
#
# print("======================LR======================")
#
# parameters = {'C': [0.1, 1.0, 10.0],
#               'solver': ["liblinear", "lbfgs", "sag"],
#               'max_iter': [50, 100, 200]}
#
# logisticRegr = LogisticRegression()
# lr_model = GridSearchCV(logisticRegr, parameters, cv=kfold)
# lr_model.fit(x_train, y_train)
#
# print('best parameter: ', lr_model.best_params_)
# print('best score: %.2f' % lr_model.best_score_)
#
# # total_param = lr_model.cv_results_['params']
# # total_score = lr_model.cv_results_["mean_test_score"]
#
# lr_best = lr_model.best_estimator_
#
# # 검증
# prediction = lr_best.predict(x_valid)
#
# print('score : {:.4f}'.format(accuracy_score(y_valid, prediction)))
# print(confusion_matrix(y_valid, prediction))
# print(classification_report(y_valid, prediction))

# ========================================== #


# # load model(저장된 거 불러와서 새로운 테스트셋 성능 검증할 때)
# load_model = pickle.load(open(name, 'rb'))
# print(len(x_valid))
# print(x_valid)
# prediction = load_model.predict(x_valid)
#
# print('score : {}'.format(round(load_model.best_estimator_.score(x_valid, y_valid), 3)))
# print(confusion_matrix(y_valid, prediction))
# print(classification_report(y_valid, prediction))