globals().clear()

####################################################
# Step 1: Import Packages, Functions, and Classes
####################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os
# %%
####################################################
# Step 2: Get Data and transform the columns
####################################################
os.chdir("/Users/adhitchandy/Library/CloudStorage/OneDrive-M365UniversitätHamburg/Semester 4/Machine Learning/Tutorial/data")

raw_data = pd.read_csv("watermelon_3_1.csv",encoding='utf-16le')
df=raw_data

print(f'The shape of the dataframe is {df.shape}')
print(df.columns.to_list())

# check for any null values
print(df[df.isnull().any(axis=1)].count())
#%%

df = df.dropna(how='any',axis=0) 

# check for class imbalance
print(df['ripe'].value_counts())

# Summary stats of data
print(df.info())
print(df.iloc[:,1:].describe())  # only summary stats for numeric columns

# %%
######################
# Feature Engineering
######################
cat_cols = df.select_dtypes('object').columns.to_list()
num_cols = df.select_dtypes('float').columns.to_list()

# filter categoricals variables from df which only contain two different values
df_cat = df.select_dtypes(include=['object'])
cat_cols_2_vals = df_cat.nunique()
cat_cols_2_vals = cat_cols_2_vals[cat_cols_2_vals == 2].index.to_list()

cat_cols = set(cat_cols) ^ set(cat_cols_2_vals)  # find not intersected elements

# extract prediction labels
labels = df['ripe'].unique().tolist()

for col in cat_cols:
    df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col)], axis=1)

# BOOLEAN TYPE FEATURES (YES /NO)
for col in cat_cols_2_vals:
    df[col + '_new'] = df[col].apply(lambda x: 1 if x == 'hard' else 0)
    df.drop(col, axis=1, inplace=True)

print(df.info())

x = df.drop(['ID','ripe'], axis=1)
cols_x=x.columns
y = np.array(df[['ripe']]).flatten() # Adjust to get 1d array
###################
# Plot the data 
###################
# plot size
plt.figure(1, figsize=(15, 8))
# plotting the data 
scatter=plt.scatter(df[['density']],df[['sugar']], c=y)#
plt.legend(handles=scatter.legend_elements()[0], labels=["False","True"])
plt.ylabel("sugar")
plt.xlabel("density")
plt.title('Watermelon 3.0')
plt.show()

##################
# Scaling data
##################
from sklearn.preprocessing import StandardScaler

# Data standardization to rescale attributes so that they have mean 0 and variance 1 -> (x -mean)/ std
# Goal: bring all features to common scale without distorting differences in the ranges of values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x[num_cols])
x = pd.concat([x.drop(columns=num_cols), pd.DataFrame(x_scaled, columns=num_cols, index=x.index)], axis=1)
X = pd.DataFrame(x, columns=cols_x)

# plot size
plt.figure(1, figsize=(15, 8))
# plotting the data 
scatter=plt.scatter(X[['density']],X[['sugar']], c=y)#
plt.legend(handles=scatter.legend_elements()[0], labels=["False","True"])
plt.ylabel("sugar")
plt.xlabel("density")
plt.title('Watermelon 3.0 scaled')
plt.show()

# %%
############################################################
# Step 3: Create a Model and Train It  (LOGISTIC REGRESSION)
############################################################

logreg = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
# solver is a string ('liblinear' by default) that decides what solver to use for fitting the model. 
# Other options are 'newton-cg', 'lbfgs', 'sag', and 'saga'.

#random_state is an integer, an instance of numpy.RandomState, or None (default) 
#that defines what pseudo-random number generator to use.

# C is a positive floating-point number (1.0 by default) that defines the relative strength of regularization. 
# Smaller values indicate stronger regularization.

logreg_result=logreg.fit(X, y)
y_pred_logreg=logreg.predict(X)

# Quick look at the attributes
print(
      "Classes: ", logreg.classes_, "\n",
      "Intercept: ", logreg.intercept_,"\n",
      "Coefficients: ", logreg.coef_
      )

#%%
####################################################
# Step 4: Evaluate the Model
####################################################

print(f'Accuracy: {accuracy_score(y, y_pred_logreg)}')
print(f'Confusion Matrix: {confusion_matrix(y, y_pred_logreg)}')
print(f'Classification Report: {classification_report(y, y_pred_logreg)}')

print(logreg.predict_proba(X)) # Predicted Output False/True 
print(logreg.predict(X)) #Actual Predictions
#%%
# Cofusion matrix as heat map 
cm = confusion_matrix(y, y_pred_logreg)
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted False', 'Predicted True'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual False', 'Actual Ture'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title('Confusion matrix for Logistic Regression for the Watermelon 3.0 data')
plt.show()


from sklearn.metrics import roc_curve, roc_auc_score

logreg_roc_auc = roc_auc_score(y, logreg.predict(X))
fpr, tpr, thresholds = roc_curve(y, logreg.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logreg_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
####################################################
# Step 5: Improve the Model
####################################################
logreg2 = LogisticRegression(solver='liblinear', C=1, random_state=0)

logreg2.fit(X, y)

print(
      "Classes: ", logreg2.classes_, "\n",
      "Intercept: ", logreg2.intercept_,"\n",
      "Coefficients: ", logreg2.coef_
      )
print(confusion_matrix(y, logreg2.predict(X)))
print(classification_report(y, logreg2.predict(X)))

#%%
##################################################
# Step 6: Plot the model  
#################################################

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(np.arange(17), y, color="black", zorder=20, marker="s", s=25)
x_test = np.linspace(-5, 10, 300)

plt.scatter(np.arange(17), logreg.predict(X), color="red", zorder=20,s=15)

plt.ylabel("y")
plt.xlabel("ID")
plt.yticks([0, 1])
#plt.yticklabels([False,  True], minor=False)
plt.ylim(-0.25, 1.5)
plt.legend(
    ("Actual value", "Predicted value"),
    loc="upper left",
    fontsize="small",
)
plt.tight_layout()
plt.title('Logistic Regression for the Watermelon 3.0 data')
plt.show()

#%%
######################################################################
# Step 3: Create a Model and Train It  (LINEAR DISCRIMINANT ANALYSIS)
######################################################################
lda = LinearDiscriminantAnalysis()
# solver is a string ('svd' by default) that decides what solver to use for fitting the model. 
# ‘svd’: Singular value decomposition (default). Does not compute the covariance matrix, 
#           therefore this solver is recommended for data with a large number of features.

lda_result=lda.fit(X, y)
y_pred_lda = lda_result.predict(X)

# Quick look at the attributes
print(
      "Classes: ", lda.classes_, "\n",
      "Intercept: ", lda.intercept_,"\n",
      "Coefficients: ", lda.coef_
      )

#%%
####################################################
# Step 4: Evaluate the Model
####################################################

print(f'Accuracy: {accuracy_score(y, y_pred_lda)}')
print(f'Confusion Matrix: {confusion_matrix(y, y_pred_lda)}')
print(f'Classification Report: {classification_report(y, y_pred_lda)}')

print(lda.predict_proba(X)) # Predicted Output False/True 
print(lda.predict(X)) #Actual Predictions

# Cofusion matrix as heat map 
cm = confusion_matrix(y, y_pred_lda)
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted False', 'Predicted True'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual False', 'Actual Ture'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title('Confusion matrix for LDA for the Watermelon 3.0 data')
plt.show()


lda_roc_auc = roc_auc_score(y, lda.predict(X))
fpr, tpr, thresholds = roc_curve(y, lda.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='LDA (area = %0.2f)' % lda_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


#%%
####################################################
# Step 5: Improve the Model
####################################################
lda2 = LinearDiscriminantAnalysis(solver='lsqr')
lda2.fit(X, y)

print(
      "Classes: ", lda2.classes_, "\n",
      "Intercept: ", lda2.intercept_,"\n",
      "Coefficients: ", lda2.coef_
      )
print(confusion_matrix(y, lda2.predict(X)))
print(classification_report(y, lda2.predict(X)))

#%%
##################################################
# Step 6: Plot the model  
#################################################

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(np.arange(17), y, color="black", zorder=20, marker="s", s=25)
x_test = np.linspace(-5, 10, 300)

plt.scatter(np.arange(17), lda.predict(X), color="red", zorder=20,s=15)

plt.ylabel("y")
plt.xlabel("ID")
plt.yticks([0, 1])
#plt.yticklabels([False,  True], minor=False)
plt.ylim(-0.25, 1.5)
plt.legend(
    ("Actual value", "Predicted value"),
    loc="upper left",
    fontsize="small",
)
plt.title('LDA for the Watermelon 3.0 data')
plt.tight_layout()
plt.show()




































