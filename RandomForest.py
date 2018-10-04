#Implement random forest classification for feature selection
#We also want to see if accounting for nonlinearity in the data will help benefit the model

#Also we can filter some features and see if it benefits Logistic Regression

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Change directory to where the data is
import os
os.chdir("../chartlabdata-any")

#Import the data to be fit
wholedata=pickle.load(open("whole-dataset.pkl",'rb'))

#Set up data to be fit by LogReg
y = wholedata["expire"]

#Specify all columns of data other than subject_id and expire flag
X=wholedata.iloc[:,2:]

# Splitting the dataset randomly into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Import Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 12324)

#Fit the RF classifier
rf.fit(X_train, y_train)

# Use the forest's predict method on the test data
y_predict = rf.predict_proba(X_test)[:,1]

#Calculate model metrics
from sklearn.metrics import confusion_matrix
decision_cutoffs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
print("Calculating sensitivity (true pos rate), specificity/selectivity (true neg rate), false pos rate, false neg rate, and accuracy")
for cutoff in decision_cutoffs:
    tn, fp, fn, tp = confusion_matrix(y_test, (y_predict>cutoff)*1).ravel()
    p=fn+tp
    n=fp+tn
    tpr=tp/p
    tnr=tn/n
    fpr=1-tnr
    fnr=1-tpr
    acc=(tp+tn)/(p+n)
    print("For cutoff of ",cutoff,': ',tpr,' ',tnr,' ',fpr,' ',fnr,' ',acc)

#Calculate AUROC and display curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
rf_roc_auc = rf.score(X_test, y_test)
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC-RF.png', format='png')
plt.show()

#Calculate AUROC
auroc=roc_auc_score(y_test, y_predict)
print("The area under the ROC curve for the test set is = ",auroc)

y_trainpredict = rf.predict(X_train)
auroc=roc_auc_score(y_train, y_trainpredict)
print("The area under the ROC curve for the training set is = ",auroc)

#Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_list = list(X.columns)
feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#Try dropping a fraction of the features with the lowest importance and fitting again
ranked_features=list()
for feature_and_importance in feature_importances:
    ranked_features.append(feature_and_importance[0])

#Specify the fraction of the features to drop
drop_fraction=0.75

#Drop them
X=wholedata.iloc[:,2:].drop(columns=ranked_features[int(len(ranked_features)*(1-drop_fraction)//1):])

#Get numerical feature importances
importances = list(rf.feature_importances_)

# List the remaining features and their importances
feature_list = list(X.columns)
feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

pickle.dump(pickle_jar, open("smallerfeatures.pkl",'wb') )