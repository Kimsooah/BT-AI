import numpy as np
import pandas as pd

train = pd.read_csv("./data/cs-training.csv")
print(train)

# kiem tra du lieu bi mat
train.isnull().sum()
print(train.isnull().sum())

train = train.drop(train.columns.values[0], axis=1)
print(train.shape)
print(train.head())
print(train.columns.values)

data_features = pd.read_excel("./data/Data Dictionary.xls")
print(data_features)

# xoa du lieu bi thieu
train_no_missing = train.dropna()
print(train_no_missing.shape)

#
train_imputer = train
train_imputer['MonthlyIncome'].fillna(train_imputer['MonthlyIncome'].mean(), inplace=True)
train_imputer['NumberOfDependents'].fillna(train_imputer['NumberOfDependents'].mean(), inplace=True)
print(train_imputer.shape)

# du lieu huan luyen
x_train = train_no_missing.drop(train_no_missing.columns.values[0], axis=1)
print(x_train)
y_train = train_no_missing[train.columns.values[0]]
print(y_train)

# du lieu kiem tra
test = pd.read_csv('./data/cs-test.csv')
test = test.drop(test.columns.values[0], axis=1)
print(test.shape)
print(test.head())
x_test = test.drop(test.columns.values[0], axis=1)
print(x_test.shape)
y_test = test[test.columns.values[0]]
print(y_test.shape)

# kiem tra du lieu bi mat(missing)
print(x_test.isnull().sum())

# bo qua cac ban ghi khong co gia tri
x_test_no_missing = x_test.dropna()
print(x_test_no_missing.shape)

# khoi phuc ban ghi khong co gia tri
x_test_imputer = x_test
x_test_imputer['MonthlyIncome'].fillna(x_test_imputer['MonthlyIncome'].mean(), inplace=True)
x_test_imputer['NumberOfDependents'].fillna(x_test_imputer['NumberOfDependents'].mean(), inplace=True)
print(x_test_imputer.isnull().sum())

# doi voi du lieu loai bo cac ban ghi bi loi
print(x_train.shape, y_train.shape, x_test_no_missing.shape)

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

classifier = DecisionTreeClassifier(max_depth=3) #max_depth=10
classifier.fit(x_train, y_train)
print(classifier.fit(x_train, y_train))
features = x_train.columns.values

from sklearn.tree import export_graphviz
export_graphviz(classifier, out_file='tree_credit.dot', feature_names=features)

from subprocess import call
call('dot -Tpng tree_credit.dot -o tree_credit.png')

from PIL import Image
Image.open('tree_credit.png').show()

import matplotlib.pyplot as plt
n = len(features)
plt.figure(figsize = (8,10))
plt.barh(range(n), classifier.feature_importances_)
plt.yticks(range(n), features)
plt.title('Muc do quan trong cac thuoc tinh')
plt.ylabel('Cac thuoc tinh')
plt.xlabel('Muc do')
plt.show()

y_predict = classifier.predict(x_test_no_missing)
print(y_predict[:5])

plt.hist(y_predict, bins=10)
plt.show()

print(classifier.score(x_train, y_train))

# doi voi du lieu imputer
print(train_imputer.shape, x_test_imputer.shape)

x_train = train_imputer.drop(train_imputer.columns.values[0], axis=1)
y_train = train_imputer[train_imputer.columns.values[0]]
dt_classifier = DecisionTreeClassifier(max_depth=25) #max_depth=25
dt_classifier.fit(x_train, y_train)
export_graphviz(dt_classifier, out_file='tree_credit_imputer.dot', feature_names=features)
# call('dot -Tpng tree_credit_imputer.dot -o tree_credit_imputer.png')
Image.open('tree_credit_imputer.png').show()

y_pred_imputer = dt_classifier.predict(x_test_imputer)
plt.hist(y_pred_imputer, bins=10)
plt.show()

print(dt_classifier.score(x_train, y_train))

### Danh gia ket qua
data_label_no_missing = train_no_missing
print(data_label_no_missing.shape)
data_no_missing = data_label_no_missing.drop(train_no_missing.columns.values[0],axis=1)
label_no_missing = data_label_no_missing[data_label_no_missing.columns.values[0]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_no_missing, label_no_missing, test_size=0.2,
                                                    stratify=label_no_missing, random_state=42)
# tree_classifier = DecisionTreeClassifier(max_depth=10)
tree_classifier = DecisionTreeClassifier(max_depth=25)
tree_classifier.fit(x_train, y_train)

from sklearn.metrics import roc_auc_score
y_train_prob_pred = tree_classifier.predict_proba(x_train)
y_test_prob_pred = tree_classifier.predict_proba(x_test)
#print('auc on training data: {:.4f}'.format(roc_auc_score(y_train, y_train_prob_pred[:,1])))
#print('auc on test data: {:.4f}'.format(roc_auc_score(y_test, y_test_prob_pred[:,1])))

from sklearn.tree import DecisionTreeRegressor
# tree_regress = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_regress = DecisionTreeRegressor(random_state=42)
tree_regress.fit(x_train, y_train)

# print('Độ chính xác tập huấn luyện: {:.4f}'.format(tree_regress.score(x_train, y_train)))
# print('Độ chính xác tập kiểm tra: {:.4f}'.format(tree_regress.score(x_test, y_test)))
from sklearn.metrics import roc_auc_score
y_train_pred = tree_classifier.predict(x_train)
y_test_pred = tree_classifier.predict(x_test)
# print('auc on training data: {:.4f}'.format(roc_auc_score(y_train, y_train_pred)))
# print('auc on test data: {:.4f}'.format(roc_auc_score(y_test, y_test_pred)))


# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

sns.set(style='white', context='notebook', palette='deep')
pd.options.display.max_columns = 100

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(x_train, y_train)

features = pd.DataFrame()
features['feature'] = x_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))

print('Độ chính xác tập huấn luyện: {:.4f}'.format(clf.score(x_train, y_train)))
print('Độ chính xác tập kiểm tra: {:.4f}'.format(clf.score(x_test, y_test)))
from sklearn.metrics import roc_auc_score
y_train_prob_pred = clf.predict_proba(x_train)[:, 1]
y_test_prob_pred = clf.predict_proba(x_test)[:, 1]
# print('auc on training data: {:.4f}'.format(roc_auc_score(y_train, y_train_prob_pred)))
# print('auc on test data: {:.4f}'.format(roc_auc_score(y_test, y_test_prob_pred)))

parameters = {'n_estimators': 1000, 'random_state': 20}

model = RandomForestClassifier(**parameters)
model.fit(x_train, y_train)

print('Độ chính xác tập huấn luyện: {:.4f}'.format(model.score(x_train, y_train)))
print('Độ chính xác tập kiểm tra: {:.4f}'.format(model.score(x_test, y_test)))
from sklearn.metrics import roc_auc_score
y_train_prob_pred = model.predict_proba(x_train)[:, 1]
y_test_prob_pred = model.predict_proba(x_test)[:, 1]
# print('auc on training data: {:.4f}'.format(roc_auc_score(y_train, y_train_prob_pred)))
# print('auc on test data: {:.4f}'.format(roc_auc_score(y_test, y_test_prob_pred)))

from sklearn.linear_model import LogisticRegression
logis_model = LogisticRegression(C=0.1, penalty='l1', tol=1e-6)
logis_model.fit(x_train, y_train)

print('Độ chính xác tập huấn luyện: {:.4f}'.format(logis_model.score(x_train, y_train)))
print('Độ chính xác tập kiểm tra: {:.4f}'.format(logis_model.score(x_test, y_test)))
from sklearn.metrics import roc_auc_score
y_train_prob_pred = logis_model.predict_proba(x_train)[:, 1]
y_test_prob_pred = logis_model.predict_proba(x_test)[:, 1]
print('auc on training data: {:.4f}'.format(roc_auc_score(y_train, y_train_prob_pred)))
# print('auc on test data: {:.4f}'.format(roc_auc_score(y_test, y_test_prob_pred)))


#237
import pandas as pd
import numpy as np

def transform_data(x):
    x['UnknownNumberOfDependents'] = pd.isna(x['NumberOfDependents']).astype(int)
    x['UnknownMonthlyIncome'] = pd.isna(x['MonthlyIncome']).astype(int)

    x['NoDependents'] = (x['NumberOfDependents'] == 0).astype(int)
    x['NoDependents'].loc[pd.isna(x['NoDependents'])] = 0

    x['NumberOfDependents'].loc[x['UnknownNumberOfDependents'] == 1] = 0

    x['NoIncome'] = (x['MonthlyIncome'] == 0).astype(int)
    x['NoIncome'].loc[pd.isna(x['NoIncome'])] = 0

    x['MonthlyIncome'].loc[x['UnknownMonthlyIncome'] == 1] = 0

    x['ZeroDebtRatio'] = (x['DebtRatio'] == 0).astype(int)
    x['UnknownIncomeDebtRatio'] = x['DebtRatio'].astype(int)
    x['UnknownIncomeDebtRatio'].loc[x['UnknownMonthlyIncome'] == 0] = 0
    x['DebtRatio'].loc[x['UnknownMonthlyIncome'] == 1] = 0

    x['WeirdRevolvingUtilization'] = x['RevolvingUtilizationOfUnsecuredLines']
    x['WeirdRevolvingUtilization'].loc[~(np.log(x['RevolvingUtilizationOfUnsecuredLines']) > 3)] = 0
    x['ZeroRevolvingUtilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] == 0).astype(int)
    x['RevolvingUtilizationOfUnsecuredLines'].loc[np.log(x['RevolvingUtilizationOfUnsecuredLines']) > 3] = 0

    x['Log.Debt'] = np.log(np.maximum(x['MonthlyIncome'], np.repeat(1, x.shape[0])) * x['DebtRatio'])
    x['Log.Debt'].loc[~np.isfinite(x['Log.Debt'])] = 0

    x['RevolvingLines'] = x['NumberOfOpenCreditLinesAndLoans'] - x['NumberRealEstateLoansOrLines']

    x['HasRevolvingLines'] = (x['RevolvingLines'] > 0).astype(int)
    x['HasRealEstateLoans'] = (x['NumberRealEstateLoansOrLines'] > 0).astype(int)
    x['HasMultipleRealEstateLoans'] = (x['NumberRealEstateLoansOrLines'] > 2).astype(int)
    x['EligibleSS'] = (x['age'] >= 60).astype(int)
    x['DTIOver33'] = ((x['NoIncome'] == 0) & (x['DebtRatio'] > 0.33)).astype(int)
    x['DTIOver43'] = ((x['NoIncome'] == 0) & (x['DebtRatio'] > 0.43)).astype(int)
    x['DisposableIncome'] = (1 - x['DebtRatio'])*x['MonthlyIncome']
    x['DisposableIncome'].loc[x['NoIncome'] == 1] = 0

    x['RevolvingToRealEstate'] = x['RevolvingLines'] / (1 + x['NumberRealEstateLoansOrLines'])

    x['NumberOfTime30-59DaysPastDueNotWorseLarge'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] > 90).astype(int)
    x['NumberOfTime30-59DaysPastDueNotWorse96'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] == 96).astype(int)
    x['NumberOfTime30-59DaysPastDueNotWorse98'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] == 98).astype(int)
    x['Never30-59DaysPastDueNotWorse'] = (x['NumberOfTime30-59DaysPastDueNotWorse'] == 0).astype(int)
    x['NumberOfTime30-59DaysPastDueNotWorse'].loc[x['NumberOfTime30-59DaysPastDueNotWorse'] > 90] = 0

    x['NumberOfTime60-89DaysPastDueNotWorseLarge'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] > 90).astype(int)
    x['NumberOfTime60-89DaysPastDueNotWorse96'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] == 96).astype(int)
    x['NumberOfTime60-89DaysPastDueNotWorse98'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] == 98).astype(int)
    x['Never60-89DaysPastDueNotWorse'] = (x['NumberOfTime60-89DaysPastDueNotWorse'] == 0).astype(int)
    x['NumberOfTime60-89DaysPastDueNotWorse'].loc[x['NumberOfTime60-89DaysPastDueNotWorse'] > 90] = 0

    x['NumberOfTimes90DaysLateLarge'] = (x['NumberOfTimes90DaysLate'] > 90).astype(int)
    x['NumberOfTimes90DaysLate96'] = (x['NumberOfTimes90DaysLate'] == 96).astype(int)
    x['NumberOfTimes90DaysLate98'] = (x['NumberOfTimes90DaysLate'] == 98).astype(int)
    x['Never90DaysLate'] = (x['NumberOfTimes90DaysLate'] == 0).astype(int)
    x['NumberOfTimes90DaysLate'].loc[x['NumberOfTimes90DaysLate'] > 90] = 0

    x['IncomeDivBy10'] = ((x['MonthlyIncome'] % 10) == 0).astype(int)
    x['IncomeDivBy100'] = ((x['MonthlyIncome'] % 100) == 0).astype(int)
    x['IncomeDivBy1000'] = ((x['MonthlyIncome'] % 1000) == 0).astype(int)
    x['IncomeDivBy5000'] = ((x['MonthlyIncome'] % 5000) == 0).astype(int)
    x['Weird0999Utilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] == 0.9999998999999999).astype(int)
    x['FullUtilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] == 1).astype(int)
    x['ExcessUtilization'] = (x['RevolvingUtilizationOfUnsecuredLines'] > 1).astype(int)

    x['NumberOfTime30-89DaysPastDueNotWorse'] = x['NumberOfTime30-59DaysPastDueNotWorse'] + x['NumberOfTime60-89DaysPastDueNotWorse']
    x['Never30-89DaysPastDueNotWorse'] = x['Never60-89DaysPastDueNotWorse'] * x['Never30-59DaysPastDueNotWorse']

    x['NumberOfTimesPastDue'] = x['NumberOfTime30-59DaysPastDueNotWorse'] + x['NumberOfTime60-89DaysPastDueNotWorse'] + x['NumberOfTimes90DaysLate']
    x['NeverPastDue'] = x['Never90DaysLate'] * x['Never60-89DaysPastDueNotWorse'] * x['Never30-59DaysPastDueNotWorse']
    x['Log.RevolvingUtilizationTimesLines'] = np.log1p(x['RevolvingLines'] * x['RevolvingUtilizationOfUnsecuredLines'])

    x['Log.RevolvingUtilizationOfUnsecuredLines'] = np.log(x['RevolvingUtilizationOfUnsecuredLines'])
    x['Log.RevolvingUtilizationOfUnsecuredLines'].loc[pd.isna(x['Log.RevolvingUtilizationOfUnsecuredLines'])] = 0
    x['Log.RevolvingUtilizationOfUnsecuredLines'].loc[~np.isfinite(x['Log.RevolvingUtilizationOfUnsecuredLines'])] = 0
    x = x.drop('RevolvingUtilizationOfUnsecuredLines', axis=1)

    x['DelinquenciesPerLine'] = x['NumberOfTimesPastDue'] / x['NumberOfOpenCreditLinesAndLoans']
    x['DelinquenciesPerLine'].loc[x['NumberOfOpenCreditLinesAndLoans'] == 0] = 0
    x['MajorDelinquenciesPerLine'] = x['NumberOfTimes90DaysLate'] / x['NumberOfOpenCreditLinesAndLoans']
    x['MajorDelinquenciesPerLine'].loc[x['NumberOfOpenCreditLinesAndLoans'] == 0] = 0
    x['MinorDelinquenciesPerLine'] = x['NumberOfTime30-89DaysPastDueNotWorse'] / x['NumberOfOpenCreditLinesAndLoans']
    x['MinorDelinquenciesPerLine'].loc[x['NumberOfOpenCreditLinesAndLoans'] == 0] = 0

    # Now delinquencies per revolving
    x['DelinquenciesPerRevolvingLine'] = x['NumberOfTimesPastDue'] / x['RevolvingLines']
    x['DelinquenciesPerRevolvingLine'].loc[x['RevolvingLines'] == 0] = 0
    x['MajorDelinquenciesPerRevolvingLine'] = x['NumberOfTimes90DaysLate'] / x['RevolvingLines']
    x['MajorDelinquenciesPerRevolvingLine'].loc[x['RevolvingLines'] == 0] = 0
    x['MinorDelinquenciesPerRevolvingLine'] = x['NumberOfTime30-89DaysPastDueNotWorse'] / x['RevolvingLines']
    x['MinorDelinquenciesPerRevolvingLine'].loc[x['RevolvingLines'] == 0] = 0

    x['Log.DebtPerLine'] = x['Log.Debt'] - np.log1p(x['NumberOfOpenCreditLinesAndLoans'])
    x['Log.DebtPerRealEstateLine'] = x['Log.Debt'] - np.log1p(x['NumberRealEstateLoansOrLines'])
    x['Log.DebtPerPerson'] = x['Log.Debt'] - np.log1p(x['NumberOfDependents'])
    x['RevolvingLinesPerPerson'] = x['RevolvingLines'] / (1 + x['NumberOfDependents'])
    x['RealEstateLoansPerPerson'] = x['NumberRealEstateLoansOrLines'] / (1 + x['NumberOfDependents'])
    x['UnknownNumberOfDependents'] = (x['UnknownNumberOfDependents']).astype(int)
    x['YearsOfAgePerDependent'] = x['age'] / (1 + x['NumberOfDependents'])

    x['Log.MonthlyIncome'] = np.log(x['MonthlyIncome'])
    x['Log.MonthlyIncome'].loc[~np.isfinite(x['Log.MonthlyIncome']) | np.isnan(x['Log.MonthlyIncome'])] = 0
    x = x.drop('MonthlyIncome', axis=1)
    x['Log.IncomePerPerson'] = x['Log.MonthlyIncome'] - np.log1p(x['NumberOfDependents'])
    x['Log.IncomeAge'] = x['Log.MonthlyIncome'] - np.log1p(x['age'])

    x['Log.NumberOfTimesPastDue'] = np.log(x['NumberOfTimesPastDue'])
    x['Log.NumberOfTimesPastDue'].loc[~np.isfinite(x['Log.NumberOfTimesPastDue'])] = 0

    x['Log.NumberOfTimes90DaysLate'] = np.log(x['NumberOfTimes90DaysLate'])
    x['Log.NumberOfTimes90DaysLate'].loc[~np.isfinite(x['Log.NumberOfTimes90DaysLate'])] = 0

    x['Log.NumberOfTime30-59DaysPastDueNotWorse'] = np.log(x['NumberOfTime30-59DaysPastDueNotWorse'])
    x['Log.NumberOfTime30-59DaysPastDueNotWorse'].loc[~np.isfinite(x['Log.NumberOfTime30-59DaysPastDueNotWorse'])] = 0

    x['Log.NumberOfTime60-89DaysPastDueNotWorse'] = np.log(x['NumberOfTime60-89DaysPastDueNotWorse'])
    x['Log.NumberOfTime60-89DaysPastDueNotWorse'].loc[~np.isfinite(x['Log.NumberOfTime60-89DaysPastDueNotWorse'])] = 0

    x['Log.Ratio90to30-59DaysLate'] = x['Log.NumberOfTimes90DaysLate'] - x['Log.NumberOfTime30-59DaysPastDueNotWorse']
    x['Log.Ratio90to60-89DaysLate'] = x['Log.NumberOfTimes90DaysLate'] - x['Log.NumberOfTime60-89DaysPastDueNotWorse']

    x['AnyOpenCreditLinesOrLoans'] = (x['NumberOfOpenCreditLinesAndLoans'] > 0).astype(int)
    x['Log.NumberOfOpenCreditLinesAndLoans'] = np.log(x['NumberOfOpenCreditLinesAndLoans'])
    x['Log.NumberOfOpenCreditLinesAndLoans'].loc[~np.isfinite(x['Log.NumberOfOpenCreditLinesAndLoans'])] = 0
    x['Log.NumberOfOpenCreditLinesAndLoansPerPerson'] = x['Log.NumberOfOpenCreditLinesAndLoans'] - np.log1p(x['NumberOfDependents'])

    x['Has.Dependents'] = (x['NumberOfDependents'] > 0).astype(int)
    x['Log.HouseholdSize'] = np.log1p(x['NumberOfDependents'])
    x = x.drop('NumberOfDependents', axis=1)

    x['Log.DebtRatio'] = np.log(x['DebtRatio'])
    x['Log.DebtRatio'].loc[~np.isfinite(x['Log.DebtRatio'])] = 0
    x = x.drop('DebtRatio', axis=1)

    x['Log.DebtPerDelinquency'] = x['Log.Debt'] - np.log1p(x['NumberOfTimesPastDue'])
    x['Log.DebtPer90DaysLate'] = x['Log.Debt'] - np.log1p(x['NumberOfTimes90DaysLate'])

    x['Log.UnknownIncomeDebtRatio'] = np.log(x['UnknownIncomeDebtRatio'])
    x['Log.UnknownIncomeDebtRatio'].loc[~np.isfinite(x['Log.UnknownIncomeDebtRatio'])] = 0
    # x['IntegralDebtRatio'] = None
    x['Log.UnknownIncomeDebtRatioPerPerson'] = x['Log.UnknownIncomeDebtRatio'] - x['Log.HouseholdSize']
    x['Log.UnknownIncomeDebtRatioPerLine'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberOfOpenCreditLinesAndLoans'])
    x['Log.UnknownIncomeDebtRatioPerRealEstateLine'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberRealEstateLoansOrLines'])
    x['Log.UnknownIncomeDebtRatioPerDelinquency'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberOfTimesPastDue'])
    x['Log.UnknownIncomeDebtRatioPer90DaysLate'] = x['Log.UnknownIncomeDebtRatio'] - np.log1p(x['NumberOfTimes90DaysLate'])

    x['Log.NumberRealEstateLoansOrLines'] = np.log(x['NumberRealEstateLoansOrLines'])
    x['Log.NumberRealEstateLoansOrLines'].loc[~np.isfinite(x['Log.NumberRealEstateLoansOrLines'])] = 0
    x = x.drop('NumberRealEstateLoansOrLines', axis=1)

    x = x.drop('NumberOfOpenCreditLinesAndLoans', axis=1)

    x = x.drop('NumberOfTimesPastDue', axis=1)
    x = x.drop('NumberOfTimes90DaysLate', axis=1)
    x = x.drop('NumberOfTime30-59DaysPastDueNotWorse', axis=1)
    x = x.drop('NumberOfTime60-89DaysPastDueNotWorse', axis=1)

    x['LowAge'] = (x['age'] < 18) * 1
    x['Log.age'] = np.log(x['age'] - 17)
    x['Log.age'].loc[x['LowAge'] == 1] = 0
    x = x.drop('age', axis=1)
    return x

kaggle_data = pd.read_csv("./data/cs-training.csv")
kaggle_data.drop(['Unnamed: 0'], axis=1, inplace=True)
process_data = transform_data(kaggle_data)
print(process_data.head())
