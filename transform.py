# --coding:utf-8 --
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from analyze import train_data

'''四、定性转换（Qualitative Transformation)'''
# 1.Dummy Variables
embark_dummies = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)
train_data.drop(['Embarked'], axis=1, inplace=True)
embark_dummies = train_data[['S', 'C', 'Q']]
# print(embark_dummies.head())

# 2.Factorizing
# 用U0代替空值
train_data['Cabin'][train_data.Cabin.isnull()] = 'U0'
# 为Cabin里的字母部分创建特征描述
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile('([a-zA-Z)]+)').search(x).group())
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
# print(train_data['CabinLetter'].head())

'''五、定量转换(Quantitative Transformation)'''
# 1.Scaling
assert np.size(train_data['Age']) == 891
scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
# print(train_data['Age_scaled'].head())

# 2.Binning处理后，或者factorize，或者dummies
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
# print(train_data['Fare_bin'].head())
# factorize
train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
# dummies
fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))
train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)
# print(train_data['Fare_bin'].head())

'''五、特征工程'''
# 对训练数据和测试数据进行处理，使得二者具有相同的数据类型和数据分布
# 对数据进行特征工程，就是从各项参数中提取出对输出结果有或大或小影响的特征，将这些特征作为训练模型的依据
train_df_org = pd.read_csv('train.csv')
test_df_org = pd.read_csv('test.csv')
test_df_org['Survived'] = 0
combined_train_test = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']

# 1.登船港口Embarked，因为缺失值不多，所以用众数填补
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
# 对于三种不同港口S/C/Q，可以用两种特征处理方式：dummies和factorizing
combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],
                                prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
# print(combined_train_test.head())

# 2.Sex
# factorize
combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
# dummies
sex_dummies_df = pd.get_dummies(combined_train_test['Sex'],
                                prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)
# print(combined_train_test.head())

# 3.Name
# 从名字中提取称呼
combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
# 各种称呼进行统一
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royal'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
# 使用dummy对不同称呼进行分列
# factorize
combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
# dummy
title_dummies_df = pd.get_dummies(combined_train_test['Title'],
                                  prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
# print(combined_train_test.head())
# 增加名字长度的特征
combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)

# 4.票价Fare
# 对缺失值按照1/2/3等级的座舱均价来填充。transform将函数np.mean应用到各个group中
combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
    combined_train_test.groupby('Pclass').transform(np.mean))

# 通过对Ticket数据进行分析，可以看到部分票号数据存在重复，同时结合亲属人数及名字的数据和票价船舱等级的对比，
# 可以知道票可以有家庭票和团体票，所以需要将团体票的票价分配到每个人的头上
combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform(
    'count')
combined_train_test['Fare'] /= combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
# 使用binning给票价分等级
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
# 对于5个等级的票价可以用dummy为票价等级分列
combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)


# 5.座舱等级Pclass
# 假设对于不同等级的船舱，其内部的票价说明了各等级舱的位置，那么也很有可能与逃生的顺序有关。
# 所以分出每个船舱里的高价位和低价位
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'


Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]

# 建立Pclass_Fare_Category
combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category,
                                                                        args=(Pclass1_mean_fare,
                                                                              Pclass2_mean_fare,
                                                                              Pclass3_mean_fare),
                                                                        axis=1)
pclass_level = LabelEncoder()
# 给每一项添加标签
pclass_level.fit(np.array(['Pclass1_Low',
                           'Pclass1_High',
                           'Pclass2_Low',
                           'Pclass2_High',
                           'Pclass3_Low',
                           'Pclass3_High']))
# 转换成数值
combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
# dummy转换
pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(
    columns=lambda x: 'Pclass_' + str(x))
combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)
# factorize
combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]


# 6.Parch & SibSp
# 前面分析过，没有或者有很多亲属会影响生存，所以将二者合并为FamilySize这一组合项，同时保留前两项
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'


combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])
family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],
                                        prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)
# print(combined_train_test.head())

# 7.Age
# 此项有缺失值，因此可以综合Sex/Title/Pclass等没有缺失值的项进行预测填充
# 提取可能与年龄有关的特征
missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size',
                                                   'Family_Size_Category', 'Fare', 'Fare_bin_id', 'Pclass']])
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
# print(missing_age_test.head())
# print('-'*40)
# print(missing_age_train.head())


# 建立Age的预测模型，可以进行多模型预测，然后再做模型融合，提高预测的精度。这里用Gradient boosting machine/random forest
def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

    # model 1 Gradient boosting machine
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000],
                          'max_depth': [4],
                          'learning_rate': [0.01],
                          'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg,
                                                gbm_reg_param_grid,
                                                cv=10,
                                                n_jobs=25,
                                                verbose=1,
                                                scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params: ' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score: ' + str(gbm_reg_grid.best_params_))
    print('GB Train Error for "Age" Feature Regressor: ' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])

    # model 2 Random Forest
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg,
                                               rf_reg_param_grid,
                                               cv=10,
                                               n_jobs=25,
                                               verbose=1,
                                               scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params: ' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score: ' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])

    # 合并两个模型
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])
    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)
    return missing_age_test


combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)

# 8.船票编号Ticket
# 此项有字母和数字的区别，对于不同的字母，很可能代表不同的船舱位置，从而对生存率产生影响，
# 所以对字母和数字进行区分，数字部分归为一类
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)

# 如果是提取数字信息，也可以按照相同的方法，如下将数字票单纯地分为一类
# combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: to_numeric(x, errors='coerce'))
# combined_train_test['Ticket_Number'].fillna(0, inplace=True)

combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]

# 9.Cabin
# 因为这一项缺失值太多，很难进行分析或预测，所以本可以直接将这一项去除。
# 但是通过上面的分析，该特征信息也与生存率有一定的关系，所以暂时保留，并将其分为有和无两类
combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

'''特征间相关性分析'''
# 挑选一些主要特征，生成特征图之间的关联图，查看特征之间的相关性
Correlation = pd.DataFrame(combined_train_test[['Embarked',
                                                'Sex',
                                                'Title',
                                                'Name_length',
                                                'Family_Size',
                                                'Family_Size_Category',
                                                'Fare',
                                                'Fare_bin_id',
                                                'Pclass',
                                                'Pclass_Fare_Category',
                                                'Age',
                                                'Ticket_Letter',
                                                'Cabin']])
colormap = plt.cm.viridis
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(Correlation.astype(float).corr(), linewidths=0.1,
            vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

# 特征之间的数据分布图
'''g = sns.pairplot(combined_train_test[['Survived', 'Pclass', 'Sex', 'Age', 'Fare',
                                      'Embarked', 'Family_Size', 'Title', 'Ticket_Letter']],
                 hue='Survived',
                 palette='seismic',
                 size=1.2,
                 diag_kind='kde',
                 diag_kws=dict(shade=True),
                 plot_kws=dict(s=10))

plt.show()'''


# 输入模型前的一些处理
# 1.将Age & Fare正则化
scale_age_fare = preprocessing.StandardScaler().fit_transform(combined_train_test[['Age', 'Fare', 'Name_length']])

# 2.先备份再弃掉无用特征
combined_data_backup = combined_train_test
combined_train_test.drop(['PassengerId',
                          'Embarked',
                          'Sex',
                          'Name',
                          'Title',
                          'Fare_bin_id',
                          'Pclass_Fare_Category',
                          'Parch',
                          'SibSp',
                          'Family_Size_Category',
                          'Ticket'],
                         axis=1,
                         inplace=True)
# 3.将训练数据和测试数据分开
train_data = combined_train_test[:891]
test_data = combined_train_test[891:]

titanic_train_data_X = train_data.drop(['Survived'], axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(["Survived"], axis=1)
# print(titanic_train_data_X.shape)
