#The goal of this script is two-fold:
#First, we would like to remove patients from the data frame that are missing more than 20% of data
#Next, for patients missing some sparse amounts of data, fill in these values by averaging over the other patients data

#Lets load some libraries
import pickle
import os
import numpy as np
import pandas as pd
import statistics as st

#Change directory to where the data is
os.chdir("../chartlabdata-any")

#Load the data frame
dataframe=pickle.load(open("float-patient-vectors.pkl",'rb'))[0]

#Get a list of the patient ids
patients=list(dataframe.patient)

#Look through all patients and assess how much nan's or "empty" elements they have
for patient in patients:
    number_empty=np.array((dataframe[dataframe.patient==patient]=="empty")*1).sum()
    number_nan=np.array(pd.isna(dataframe[dataframe.patient==patient])*1).sum()
    fraction_bad=(number_empty+number_nan)/int(dataframe.shape[1])
    
    if fraction_bad>0.2:
        dataframe=dataframe[dataframe.patient!=patient]
        
#Look through all columns and replace "empty" and nan with averaged values
#Get a list of the column labels in dataframe

dataframe=dataframe.fillna(0)

for patient in patients:
    number_zero=np.array((dataframe[dataframe.patient==patient]==0)*1).sum()
    fraction_bad=(number_zero)/int(dataframe.shape[1])
    if fraction_bad>0.2:
        dataframe=dataframe[dataframe.patient!=patient]
    
#Pickle the vectors file
pickle_jar=[dataframe]
pickle.dump(pickle_jar, open("clean-vectors.pkl",'wb') )

    
