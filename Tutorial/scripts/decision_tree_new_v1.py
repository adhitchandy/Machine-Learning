# import packages
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
os.chdir("/Users/adhitchandy/Library/CloudStorage/OneDrive-M365UniversitätHamburg/Semester 4/Machine Learning/Tutorial/data")

raw_data = pd.read_csv("bank_marketing.csv")
df = raw_data

#Aim: Predict if client will subcribe a term deposit.

# %%
############################################################
# Data Summary
############################################################
"""

Categorical Variables :

[1] job : admin,technician, services, management, retired, blue-collar, unemployed, entrepreneur, housemaid, unknown, self-employed, student
[2] marital : married, single, divorced
[3] education: secondary, tertiary, primary, unknown
[4] default : yes, no
[5] housing : yes, no
[6] loan : yes, no
[7] deposit : yes, no (Dependent Variable)
[8] contact : unknown, cellular, telephone
[9] month : jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
[10] poutcome: unknown, other, failure, success
------------------------------------------------------------------------------------------------------------------------
Numerical Variables:
[1] age
[2] balance
[3] day
[4] duration
[5] campaign
[6] pdays
[7] previous
"""
# %%
print(f'The shape of the dataframe is {df.shape}')
print(df.columns.to_list())
# %%
# check for any null values
print(df[df.isnull().any(axis=1)].count())
# %%
# Summary stats of data
print(df.info())
sum_stats = df.describe()  # only summary stats for numeric columns

# %%
########################################################################################################################
# Feature Engineering
########################################################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from typing import Union

# select categorical data of type object
cat_cols = df.select_dtypes('object').columns.to_list()

# filter categoricals variables from df which only contain two different values
df_cat = df.select_dtypes(include=['object'])
cat_cols_2_vals = df_cat.nunique()
cat_cols_2_vals = cat_cols_2_vals[cat_cols_2_vals == 2].index.to_list()

cat_cols = set(cat_cols) ^ set(cat_cols_2_vals)  # find not intersected elements -> columns containng more than two values
# %%
# extract prediction labels
labels = df['deposit'].unique().tolist()
# %%

for col in cat_cols:
    df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col)], axis=1)

print(df.info())

# BOOLEAN TYPE FEATURES (YES /NO)

for col in cat_cols_2_vals:
    df[col + '_new'] = df[col].apply(lambda x: 1 if x == 'yes' else 0)
    df.drop(col, axis=1, inplace=True)
# %%
# Splitting data into train and test set
from sklearn.model_selection import train_test_split

X = df.drop('deposit_new', axis=1)
y = df['deposit_new']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test))
# %%
# Scaling data
from sklearn.preprocessing import StandardScaler

# Data standardization to rescale attributes so that they have mean 0 and variance 1 -> (x -mean)/ std
# Goal: bring all features to common scale without distorting differences in the ranges of values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Why do we use fit_transform on the training set and transform on the test set?
# fit_transform: learns the parameters μ and σ and transforms the data using the learned parameters.
# transform: uses the learned parameters μ and σ to transform the data. We do not want a biased model and that the model
# sees completely new data.
# %%
# Modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

"""
Instanciate the Decision Tree Classifier and fit the model
"""

clf = DecisionTreeClassifier(max_depth=2,
                             criterion="gini",  # can also be entropy or log Loss
                             splitter="best",
                             # decide which feature to split -> uses the feature with highest feature importance
                             random_state=42)
clf.fit(X_train, y_train)

# %%

"""
We have multiple options to visualize a decision tree
1. Plot a tree diagram
2. Export the tree to a text file
"""

plt.figure(figsize=(30, 10), facecolor='white')
# create the tree plot
a = tree.plot_tree(clf,
                   # use the feature names stored
                   feature_names=X.columns.tolist(),
                   # use the class names stored
                   class_names=labels,
                   rounded=True,
                   filled=True,
                   fontsize=14)

# show the plot
plt.show()

# %%
from sklearn.tree import export_text

tree_rules = export_text(clf, feature_names=list(X.columns))
# print the result
print(tree_rules)

# %%
import seaborn as sns
from sklearn import metrics

"""
Now we are interested on how the fitted tree performs on the unseen data -> test set
One way is to use the confusion matrix 
"""

# make predictions on the test set
test_pred_decision_tree = clf.predict(X_test)

# get the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test,
                                            test_pred_decision_tree)
# turn this into a dataframe
matrix_df = pd.DataFrame(confusion_matrix)
# plot the result
ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10, 7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
# set axis titles
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize=15)
ax.set_xticklabels(list(labels))  # TODO: adjust that
ax.set_ylabel("True Label", fontsize=15)
ax.set_yticklabels(list(labels), rotation=0)
plt.show()

# %%
"""
Measuring performance of the model using Accuracy, Precision and the F1 score
"""

# Fraction of TPs and TNs over the total number if assigned labels.
# Sum(diagonal of the confusion matrix) / Sum(confusion matrix)
# metrics.accuracy_score(y_test, test_pred_decision_tree)
# Accuracy = (627 + 916) / (627 + 916 +  494 + 151) = 0.71115
print(f'Accuracy: {metrics.accuracy_score(y_test, test_pred_decision_tree)}')

# How many of the values we predicted to be in a certain class are actually in that class
# True positive (number in diagonal) / All positives (column sum)
# e.g. 672/(672 + 151) = 0.8165 for 'yes'
precision = metrics.precision_score(y_test,
                                    test_pred_decision_tree,
                                    average=None)  # Can be used for multiclass classification -> e.g. micro or macro

# print(f'Precision: {precision}')
# turn it into a dataframe
precision_results = pd.DataFrame(precision, index=labels)
# rename the results column
precision_results.rename(columns={0: 'precision'}, inplace=True)
precision_results
# We have a precision of 81.65 percent for yes and 64.96 percent for no.

# %%
# How many of the values in each class were given correct label -> tells us how it perfomed relative to false negative
# True positive (number in diagonal) / All positives (row sum)
# e.g. 672/(672 + 494) = 0.5764 for 'yes'  and 916/(916 + 151) = 0.8585 for 'no'
recall = metrics.recall_score(y_test, test_pred_decision_tree,
                              average=None)
recall_results = pd.DataFrame(recall, index=labels)
recall_results.rename(columns={0: 'Recall'}, inplace=True)
recall_results
# %%
# Weighted average of precision and recall.
# 1 is the best and 0 is the worst
# harmonic mean -> prevents overerstimation of the performance of the model in cases where one parameter is high and the
# is low
# 2*(precision*recall)/(precision+recall) -> 2*(0.8165 * 0.5763)/(0.8165 + 0.5763) = 0.6757
f1 = metrics.f1_score(y_test, test_pred_decision_tree, average=None)
f1_results = pd.DataFrame(f1, index=labels)

f1_results.rename(columns={0: 'f1'}, inplace=True)
f1_results
# %%
# All metrics in one report
print(metrics.classification_report(y_test,
                                    test_pred_decision_tree))

########################################################################################################################
########################################################################################################################
# %%
########################################################################################################################
# Can we do better?
########################################################################################################################

# Try different models

dt3 = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
dt3.fit(X_train, y_train)

test_pred_decision_tree = dt3.predict(X_test)

print(metrics.classification_report(y_test,
                                    test_pred_decision_tree))

# get the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test,
                                            test_pred_decision_tree)
# turn this into a dataframe
matrix_df = pd.DataFrame(confusion_matrix)
# plot the result
ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10, 7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
# set axis titles
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize=15)
ax.set_xticklabels(list(labels))
ax.set_ylabel("True Label", fontsize=15)
ax.set_yticklabels(list(labels), rotation=0)
plt.show()

# We observe by using a deeper tree we can improve the performance of the model e.g. by means of an higher f1 score.


# %%
dt3 = tree.DecisionTreeClassifier(max_depth=10, random_state=42)
dt3.fit(X_train, y_train)

test_pred_decision_tree = dt3.predict(X_test)

# get the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test,
                                            test_pred_decision_tree)
# turn this into a dataframe
matrix_df = pd.DataFrame(confusion_matrix)
# plot the result
ax = plt.axes()
sns.set(font_scale=1.3)
plt.figure(figsize=(10, 7))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
# set axis titles
ax.set_title('Confusion Matrix - Decision Tree')
ax.set_xlabel("Predicted label", fontsize=15)
ax.set_xticklabels(list(labels))
ax.set_ylabel("True Label", fontsize=15)
ax.set_yticklabels(list(labels), rotation=0)
plt.show()

