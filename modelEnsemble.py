# --coding:utf-8 --
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transform import titanic_train_data_X, titanic_test_data_X, titanic_train_data_Y, PassengerId
from xgboost import XGBClassifier

''' 使用两层模型融合
Level 1使用：Random Forest, AdaBoost, ExtraTrees, Gradient Boosting, DecisionTree, KNN, SVM七个模型。
Level 2使用XGBoost以第一层预测的结果作为特征对最终的结果进行预测。

Stacking框架是堆叠使用基础分类器的预测作为对二级模型的训练的输入。然而，确不能简单地在全部训练数据上训练基本模型，
借以产生预测，其输出用于第二层的训练。如果在TrainData上训练，然后在TrainData上预测，就会造成标签泄露。为了避免标签泄露，
需要对每个基学习器使用K折交叉验证，将K个模型对ValidSet的预测结果拼接，作为下一层学习器的输入。'''

ntrain = titanic_train_data_X.shape[0]
ntest = titanic_test_data_X.shape[0]
SEED = 0  # 为了重复利用
NFOLDS = 7  # 为折外预测设置折数
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)


def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# 构建不同的学习框架
rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                            min_samples_split=3, min_samples_leaf=2, n_jobs=1, verbose=0)
ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3,
                                min_samples_leaf=2, max_depth=5, verbose=0)
dt = DecisionTreeClassifier(max_depth=8)
knn = KNeighborsClassifier(n_neighbors=2)
svm = SVC(kernel='linear', C=0.025)

# 将pandas转换为arrays
x_train = titanic_train_data_X.values
x_test = titanic_test_data_X.values
y_train = titanic_train_data_Y.values

# Create our OOF train and test predictions. These base results will be used as new features
rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test)  # AdaBoost
et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test)  # Extra Trees
gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test)  # Gradient Boost
dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test)  # Decision Tree
knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test)  # KNeighbors
svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test)  # Support Vector

print("Training is complete")

# 预测并生成提交文件。利用XGBoost,使用第一层预测的结果作为特征，对最终结果进行预测
x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train,
                          dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test,
                         dt_oof_test, knn_oof_test, svm_oof_test), axis=1)
gbm = XGBClassifier(n_estimator=2000, max_depth=4, min_child_weight=2, gamma=0.9,
                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                    nthread=-1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)
StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
StackingSubmission.to_csv('StackingSubmission.csv', index=False)
