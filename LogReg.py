import pickle
import pandas as pd
import numpy as np
import math

#Import the data to be fit
picklejar = pickle.load(open("data.pkl",'rb'))  
chartlabdata=picklejar[0]

#Some of the age's are listed as 300! Delete them
chartlabdata = chartlabdata[chartlabdata.age != 300]

#Some creatinine levels are listed as nan! Delete them
chartlabdata = chartlabdata[(chartlabdata.Creatinine) < 10**6]

#Set up data to be fit by LogReg
y = chartlabdata["expire"]

#Prepare the set of features
age_norm = (chartlabdata["age"]-chartlabdata["age"].mean())/chartlabdata["age"].std()
creatinine_norm = (chartlabdata["Creatinine"]-chartlabdata["Creatinine"].mean())/chartlabdata["Creatinine"].std()

#Use one hot encoding for gender
gender_one=[]
for patientnum in range(len(age_norm)):
    if chartlabdata["gender"].iloc[patientnum]=="M":
        gender_one.append(1)
    else:
        gender_one.append(0)

#Make a vector of ones for the constant in Logistic Regression
coeffs=np.ones([len(gender_one)])

#Make these into feature list
X=np.array([chartlabdata["patient"],age_norm,creatinine_norm,gender_one,coeffs]).T

# Splitting the dataset randomly into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Importing statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Calling an instance of the logit model
logit_model=sm.Logit(y_train,X_train[:,1:])

#Fit the logit model
result=logit_model.fit(method='bfgs', maxiter = 100)

#Print the results of the fit
print(result.summary())

#Predict the outcomes of the patients in the test set
y_predict = result.predict(X_test[:,1:])

#Build a matrix for comparing results
compare = np.array([y_test,y_predict]).T

#Calculate Area Under Receiver Operating Curve (AUROC)
falseneg=[]
falsepos=[]
decisionboundaries=np.linspace(0,1,num=51)
for boundary in range(len(decisionboundaries)):
    falseneg.append(0)
    falsepos.append(0)
    for patient in range(len(compare)):
        if (compare[patient][0]==1) and (compare[patient][1]<decisionboundaries[boundary]):
            falseneg[boundary]=falseneg[boundary]+1
        elif (compare[patient][0]==0) and (compare[patient][1]>decisionboundaries[boundary]):
            falsepos[boundary]=falsepos[boundary]+1
falsenegrate=np.array(falseneg)/falseneg[-1]
falseposrate=np.array(falsepos)/falsepos[0]
trueposrate=1-falseposrate

#Make a plot of the ROC
import matplotlib as plt
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(trueposrate, falsenegrate,color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Receiver Operator Curve')
ax.grid()

#Save a copy of this plot
fig.savefig("test.png")
plt.show()

#Save the model and save results for ten people
X_ten=X_train[0:10,:]
y_ten=y_train[0:10]

pickle_jar=[result,X_ten,y_ten]
pickle.dump(pickle_jar, open("model.pkl",'wb') )

