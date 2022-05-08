# --coding:utf-8 --

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
'''一、数据总览'''
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sns.set_style('whitegrid')
train_data.head()
# train_data.info()
# print('-' * 40)
# test_data.info()

train_data['Survived'].value_counts().plot.pie(autopct='%2.2f%%')

'''二、缺失值处理'''
# 对登船港口的缺失值填充众数以作补充
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
# 对舱级的缺失值进行标志性填充
train_data['Cabin'] = train_data.Cabin.fillna('U0')
# 使用随机森林对年龄的缺失值进行线性回归预测


age_df = train_data[['Age', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']]  # 抽取与年龄有关的指标
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:, 1:]  # 选取所有行，和除第一列以外的所有列的值作为训练输入样本training input samples
Y = age_df_notnull.values[:, 0]  # 选取所有行和第一列的值作为目标值target value
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X, Y)  # 会将X转换为单精度浮点数
predictAges = RFR.predict(age_df_isnull.values[:, 1:])
train_data.loc[train_data['Age'].isnull(), ['Age']] = predictAges
# train_data.info()

'''三、分析数据关系'''
# 1. 性别Sex与是否生存Survived的关系
train_data.groupby(['Sex', 'Survived'])['Sex'].count()
train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()

# 2. 船舱等级Pclass
train_data.groupby(['Pclass', 'Survived'])['Pclass'].count()
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()
train_data[['Sex', 'Pclass', 'Survived']].groupby(['Pclass', 'Sex']).mean().plot.bar()

# 3. 年龄Age
# 分析不同等级船舱和不同性别下的年龄分布和生存的关系
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.violinplot('Pclass', 'Age', hue='Survived', data=train_data, split=True, ax=ax[0])  # 以不同颜色表示是否生存
ax[0].set_title('Pclass and Age .vs. Survived')
ax[0].set_yticks(range(0, 110, 10))

sns.violinplot('Sex', 'Age', hue='Survived', data=train_data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age .vs. Survive')
ax[1].set_yticks(range(0, 110, 10))

# 分析总体的年龄分布
plt.figure(figsize=(12, 5))
plt.subplot(121)
train_data['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train_data.boxplot(column='Age', showfliers=False)

# 不同年龄下的生存和非生存的分布
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)  # 色度hue表示是否生存
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

# 不同年龄下的平均生存率
fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
train_data['Age_int'] = train_data['Age'].astype(int)
average_age = train_data[['Age_int', 'Survived']].groupby(['Age_int'], as_index=False).mean()
sns.barplot(x='Age_int', y='Survived', data=average_age)

# 按照年龄可大致将乘客划分为儿童、青少年、青年、老年四个群体
# print(train_data['Age'].describe())
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
by_age = train_data.groupby('Age_group')['Survived'].mean()
# print(by_age)
by_age.plot(kind='bar')
# plt.show()

# 4、称呼Name
train_data['Title'] = train_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
# print(pd.crosstab(train_data['Title'], train_data['Sex']))
train_data[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()
# plt.show()

# 名字长度和生存率之间存在的关系
fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length', 'Survived']].groupby(['Name_length'], as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)
# plt.show()

# 5、有无兄弟姐妹SibSp
# 分成有和无两类
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]

plt.figure(figsize=(10, 5))
# 有兄弟姐妹的存活比例
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
plt.xlabel('sibsp')
# 无兄弟姐妹的存活比例
plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
plt.xlabel('no_sibsp')
# plt.show()

# 6.有无父母子女Parch
# 分成有和无两类
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(10, 5))
# 有父母子女的存活比例
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', "Survived"], autopct='%1.1f%%')
plt.xlabel('parch')
# 无父母子女的存活比例
plt.subplot(122)
no_parch_df["Survived"].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
plt.xlabel('no_parch')
# plt.show()

# 7.亲友SibSp & Parch的人数和存活与否的关系
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
train_data[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')
train_data['Family Size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['Family Size', 'Survived']].groupby(['Family Size']).mean().plot.bar()
# plt.show()

# 8.票价Fare
plt.figure(figsize=(10, 5))
train_data['Fare'].hist(bins=70)
# 绘制以座舱等级分类的票价箱线图
train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
# 绘制是否生存与票价均值和方差的关系
# print(train_data_fare.describe())
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]

average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)

# 9.船舱类型Cabin缺失值过多，只有204例，无法分析，因此舍弃
# 10.港口Embarked和是否存活的关系
# 泰坦尼克号途径Southampton Port、Cherbourg、Queenstown,那么越早登船的人，越有可能早下船
# 因此不容易遇到沉船事故
sns.countplot('Embarked', hue='Survived', data=train_data)
plt.title('Embarked and Survived')
sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
plt.title('Embarked and Survived rate')
# plt.show()
