# --coding:utf-8 --
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from transform import titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X

'''六、模型融合与测试'''


# 1.利用不同的模型对特征进行筛选，以选出较为重要的特征
def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params: ' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score: ' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score: ' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features form RF Classifier:')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params: ' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score: ' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score: ' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier: ')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params: ' + str(et_grid.best_params_))
    print('Top N Features Best ET Scores: ' + str(et_grid.best_score_))
    print('Top N Features ET Train Score: ' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier: ')
    print(str(features_top_n_et[:10]))

    # Gradient Boosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Param: ' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score: ' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score: ' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best DT Params: ' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score: ' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score: ' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # 融合五个模型
    feature_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et,
                               features_top_n_gb, features_top_n_dt], ignore_index=True).drop_duplicates()

    feature_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_gb, feature_imp_sorted_et,
                                    feature_imp_sorted_dt, feature_imp_sorted_ada], ignore_index=True).drop_duplicates()
    return feature_top_n, feature_importance


# 2.依据筛选出的特征构建训练集和测试集
# 太多的特征一方面会影响模型训练的速度，另一方面也可能会使得模型过拟合。
# 所以在特征太多的情况下，可以利用不同的模型对特征进行筛选，选出想要的前n个特征
feature_to_pick = 30
feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])
#print(titanic_train_data_X)
#print('-'*50)
#print(titanic_test_data_X)
