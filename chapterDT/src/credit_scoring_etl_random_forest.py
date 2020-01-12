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
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

sns.set(style='white', context='notebook', palette='deep')
pd.options.display.max_columns = 100

train = pd.read_csv("./data/cs-training.csv")
kaggle_test = pd.read_csv("./data/cs-test.csv")

print(train.head())
print(train.shape)
print(train.describe())
print(train.info())
print(train.isnull().sum())
print(kaggle_test.isnull().sum())

### Target distribution
ax = sns.countplot(x = train.SeriousDlqin2yrs ,palette="Set3")
sns.set(font_scale=1.5)
ax.set_ylim(top = 150000)
ax.set_xlabel('Financial difficulty in 2 years')
ax.set_ylabel('Frequency')
fig = plt.gcf()
fig.set_size_inches(10,5)
ax.set_ylim(top=160000)

plt.show()

### Detecting outliers
def detect_outliers(df, n, features):
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


# detect outliers from Age, SibSp , Parch and Fare
# These are the numerical features present in the dataset
Outliers_to_drop = detect_outliers(train, 2, ["RevolvingUtilizationOfUnsecuredLines",
                                              "age",
                                              "NumberOfTime30-59DaysPastDueNotWorse",
                                              "DebtRatio",
                                              "MonthlyIncome",
                                              "NumberOfOpenCreditLinesAndLoans",
                                              "NumberOfTimes90DaysLate",
                                              "NumberRealEstateLoansOrLines",
                                              "NumberOfTime60-89DaysPastDueNotWorse",
                                              "Unnamed: 0",
                                              "NumberOfDependents"])
train.loc[Outliers_to_drop]
print(train.loc[Outliers_to_drop])

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

### Merging datasets

train_len = len(train)
dataset =  pd.concat(objs=[train, kaggle_test], axis=0).reset_index(drop=True)
print(train_len)
dataset.shape

dataset = dataset.rename(columns={'Unnamed: 0': 'Unknown',
                                  'SeriousDlqin2yrs': 'Target',
                                  'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
                                  'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
                                  'DebtRatio': 'DebtRatio',
                                  'MonthlyIncome': 'MonthlyIncome',
                                  'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
                                  'NumberOfTimes90DaysLate': 'Late90',
                                  'NumberRealEstateLoansOrLines': 'PropLines',
                                  'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
                                  'NumberOfDependents': 'Deps'})

train = train.rename(columns={'Unnamed: 0': 'Unknown',
                                  'SeriousDlqin2yrs': 'Target',
                                  'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
                                  'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
                                  'DebtRatio': 'DebtRatio',
                                  'MonthlyIncome': 'MonthlyIncome',
                                  'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
                                  'NumberOfTimes90DaysLate': 'Late90',
                                  'NumberRealEstateLoansOrLines': 'PropLines',
                                  'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
                                  'NumberOfDependents': 'Deps'})

kaggle_test = kaggle_test.rename(columns={'Unnamed: 0': 'Unknown',
                                  'SeriousDlqin2yrs': 'Target',
                                  'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
                                  'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
                                  'DebtRatio': 'DebtRatio',
                                  'MonthlyIncome': 'MonthlyIncome',
                                  'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
                                  'NumberOfTimes90DaysLate': 'Late90',
                                  'NumberRealEstateLoansOrLines': 'PropLines',
                                  'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
                                  'NumberOfDependents': 'Deps'})

### Exploring variables
# Correlation matrix
g = sns.heatmap(train.corr(),annot=False, fmt = ".2f", cmap = "coolwarm")

### Exploring UnsecLines
dataset.UnsecLines.describe()
dataset.UnsecLines = pd.qcut(dataset.UnsecLines.values, 5).codes

# Explore UnsecLines feature vs Target
g  = sns.factorplot(x="UnsecLines",y="Target",data=dataset,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring Age
# Explore Age vs Survived
g = sns.FacetGrid(dataset, col='Target')
g = g.map(sns.distplot, "age")

dataset.age = pd.qcut(dataset.age.values, 5).codes

# Explore age feature vs Target
g  = sns.factorplot(x="age",y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring Late3059
# Explore UnsecLines feature vs Target
g  = sns.factorplot(x="Late3059", y="Target", data=dataset, kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

for i in range(len(dataset)):
    if dataset.Late3059[i] >= 6:
        dataset.Late3059[i] = 6

# Explore UnsecLines feature vs Target
g  = sns.factorplot(x="Late3059",y="Target",data=dataset,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring DebtRatio
# Explore Age vs Survived
g = sns.FacetGrid(dataset, col='Target')
g = g.map(sns.distplot, "DebtRatio")

dataset.DebtRatio = pd.qcut(dataset.DebtRatio.values, 5).codes

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="DebtRatio",y="Target",data=dataset,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring MonthlyIncome
dataset.MonthlyIncome.isnull().sum()

g = sns.heatmap(dataset[["MonthlyIncome","Unknown","UnsecLines","OpenCredit","PropLines"]].corr(),cmap="BrBG",annot=True)

g = sns.heatmap(dataset[["MonthlyIncome","age","DebtRatio","Deps","Target"]].corr(),cmap="BrBG",annot=True)

g = sns.heatmap(dataset[["MonthlyIncome","Late3059","Late6089","Late90"]].corr(),cmap="BrBG",annot=True)

dataset.MonthlyIncome.median()

#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset.MonthlyIncome = dataset.MonthlyIncome.fillna(dataset.MonthlyIncome.median())

dataset.MonthlyIncome = pd.qcut(dataset.MonthlyIncome.values, 5).codes

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="MonthlyIncome",y="Target",data=dataset,kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring OpenCredit
dataset.OpenCredit.describe()
dataset.OpenCredit = pd.qcut(dataset.OpenCredit.values, 5).codes

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="OpenCredit", y="Target", data=dataset, kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring Late90
dataset.Late90.describe()

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Late90", y="Target", data=dataset, kind="bar", size = 6 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

for i in range(len(dataset)):
    if dataset.Late90[i] >= 5:
        dataset.Late90[i] = 5

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Late90", y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring PropLines
dataset.PropLines.describe()

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="PropLines", y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

for i in range(len(dataset)):
    if dataset.PropLines[i] >= 6:
        dataset.PropLines[i] = 6

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="PropLines", y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring Late6089
# Explore Late6089 feature quantiles vs Target
g  = sns.factorplot(x="Late6089", y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

for i in range(len(dataset)):
    if dataset.Late6089[i] >= 3:
        dataset.Late6089[i] = 3

# Explore Late6089 feature quantiles vs Target
g = sns.factorplot(x="Late6089", y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Exploring Deps
dataset.Deps.describe()
dataset.Deps = dataset.Deps.fillna(dataset.Deps.median())
dataset.Deps.isnull().sum()

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Deps", y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

for i in range(len(dataset)):
    if dataset.Deps[i] >= 4:
        dataset.Deps[i] = 4

# Explore DebtRatio feature quantiles vs Target
g  = sns.factorplot(x="Deps",y="Target", data=dataset, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Target probability")

### Final NaN check
dataset.info()
dataset.head()
dataset = pd.get_dummies(dataset, columns = ["UnsecLines"], prefix="UnsecLines")
dataset = pd.get_dummies(dataset, columns = ["age"], prefix="age")
dataset = pd.get_dummies(dataset, columns = ["Late3059"], prefix="Late3059")
dataset = pd.get_dummies(dataset, columns = ["DebtRatio"], prefix="DebtRatio")
dataset = pd.get_dummies(dataset, columns = ["MonthlyIncome"], prefix="MonthlyIncome")
dataset = pd.get_dummies(dataset, columns = ["OpenCredit"], prefix="OpenCredit")
dataset = pd.get_dummies(dataset, columns = ["Late90"], prefix="Late90")
dataset = pd.get_dummies(dataset, columns = ["PropLines"], prefix="PropLines")
dataset = pd.get_dummies(dataset, columns = ["Late6089"], prefix="Late6089")
dataset = pd.get_dummies(dataset, columns = ["Deps"], prefix="Deps")

dataset.head()
dataset.shape

### Building our credit scoring model
train = dataset[:train_len]
Kaggle_test = dataset[train_len:]
Kaggle_test.drop(labels=["Target"], axis = 1, inplace=True)

Kaggle_test.shape

## Separate train features and label
train["Target"] = train["Target"].astype(int)

Y_train = train["Target"]
X_train = train.drop(labels = ["Target", "Unknown"],axis = 1)

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X_train, Y_train)

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))

parameters = {'n_estimators': 1000, 'random_state': 20}

model = RandomForestClassifier(**parameters)
model.fit(X_train, Y_train)

Kaggle_test.head()

results_df = pd.read_csv("./data/cs-test.csv")

results_df = results_df.drop(["RevolvingUtilizationOfUnsecuredLines",
                             "age",
                             "NumberOfTime30-59DaysPastDueNotWorse",
                             "DebtRatio",
                             "MonthlyIncome",
                             "NumberOfOpenCreditLinesAndLoans",
                             "NumberOfTimes90DaysLate",
                             "NumberRealEstateLoansOrLines",
                             "NumberOfTime60-89DaysPastDueNotWorse",
                             "NumberOfDependents"], axis=1)


DefaultProba = model.predict_proba(Kaggle_test.drop(["Unknown"], axis=1))
DefaultProba = DefaultProba[:,1]
results_df.SeriousDlqin2yrs = DefaultProba

results_df = results_df.rename(columns={'Unnamed: 0': 'Id', 'SeriousDlqin2yrs': 'Probability'})

results_df.head()

results_df.to_csv("kaggle_credit_score.csv", index=False)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train,
                                                    random_state=42)

parameters = {'n_estimators': 1000, 'random_state' : 20}
rf_model = RandomForestClassifier(**parameters)
rf_model.fit(x_train, y_train)

print('Độ chính xác tập huấn luyện: {:.4f}'.format(rf_model.score(x_train, y_train)))
print('Độ chính xác tập kiểm tra: {:.4f}'.format(rf_model.score(x_test, y_test)))
from sklearn.metrics import roc_auc_score
y_train_prob_pred = rf_model.predict_proba(x_train)[:, 1]
y_test_prob_pred = rf_model.predict_proba(x_test)[:, 1]
print('auc on training data: {:.4f}'.format(roc_auc_score(y_train, y_train_prob_pred)))
print('auc on test data: {:.4f}'.format(roc_auc_score(y_test, y_test_prob_pred)))

