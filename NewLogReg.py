#Fit a new logistic regression model with fewer features
import pickle
import pandas as pd
import math
import matplotlib.pyplot as plt 

#Max number of iterations
num_iterations = 2000

#Change directory to where the data is
import os
os.chdir("../chartlabdata-any")

#Import the data to be fit
pickle_jar=pickle.load(open("smallerfeatures.pkl",'rb'))
X=pickle_jar[0]
y=pickle_jar[1]

# Splitting the dataset randomly into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Importing statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Calling an instance of the logit model
logit_model=sm.Logit(y_train,X_train)

#Fit the logit model
result=logit_model.fit(method='bfgs', maxiter = num_iterations)
print(result.summary())

y_predict = result.predict(X_test)

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

#Calculate Confusion matrix, ROC and AUROC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(max_iter = num_iterations, penalty = "l2")
logreg.fit(X_train, y_train)

y_predict = logreg.predict(X_test)

conf_matrix = confusion_matrix(y_test, (y_predict>0.5)*1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = logreg.score(X_test, y_test)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC-newlogreg.png', format='png')
plt.show()

y_predict=logreg.predict_proba(X_test)[:,1]

#Calculate AUROC
auroc=roc_auc_score(y_test, y_predict)
print("The area under the ROC curve for the test set is = ",auroc)

y_trainpredict = logreg.predict_proba(X_train)
auroc=roc_auc_score(y_train, y_trainpredict[:,1])
print("The area under the ROC curve for the training set is = ",auroc)

#Save the model and save results for ten people
X_ten=X_train.iloc[0:10,:]
y_ten=y_train.iloc[0:10]
patients=wholedata["subject_id"].iloc[0:10]
pickle_jar=[logreg,X_ten,y_ten,patients]

pickle.dump(pickle_jar, open("model.pkl",'wb') )