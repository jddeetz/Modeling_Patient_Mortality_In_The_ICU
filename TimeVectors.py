#We want to get a series of time vectors for a small set of patients.

#First we import some libraries
import pickle
import os
import statistics as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


#Change to working directory
os.chdir("../chartlabdata-any")

#First we need to load a pickle file containing the patients ids
pickle_jar=pickle.load(open("model.pkl",'rb'))
patients=pickle_jar[3]
X_ten=pickle_jar[1]

#Now that we have a list of the patients, lets define some of the parameters we will use

#Define the width of the time window to be used to generate vectors
window_width = 48 #hours

#We will also use a window stride of 1 hour, so we will generate a new vector of features every hour

#Get hadm_id, expire, age, gender admittime, dischtime
demodata=pickle.load(open("data.pkl",'rb'))[0]

#Make a blank template for each dataframe, so that each has exactly the same number of features in the same order as the model.
empty=X_ten[0:0]
patientnum=-1

patient_dataframes=dict()

#Lets loop through each patient
for patient in patients:

    patientnum=patientnum+1
    a=-1
    print("Opening chart and lab files for patient",patient)
    
    chartdata=pickle.load(open(str(patient)+"-smallerchart.pkl",'rb'))[0]
    labdata=pickle.load(open(str(patient)+"-smallerlab.pkl",'rb'))[0]
    
    #Create dictionary for feature timelines, each key will be an item of lab or chart events
    feature_timelines=dict()
    
    print("Unrolling chart and lab time series data for patient ",patient)
    
    #Assemble a list of the features and unroll data into feature timelines
    features=list()
    for line in chartdata:
        #This is what a string of chartdata looks like
        #(145576, datetime.datetime(2195, 4, 6, 8, 0), 'CVP', '11', 11.0, 'mmHg')
        #hospital admissions id, time, item, stringvalue, float value, units
        if line[2] in feature_timelines:
            feature_timelines[line[2]].append(line[4])
            feature_timelines[line[2]+"-times"].append(line[1])
        else:
            feature_timelines[line[2]]=list()
            feature_timelines[line[2]+"-times"]=list()
            feature_timelines[line[2]].append(line[4])
            feature_timelines[line[2]+"-times"].append(line[1])
            features.append(line[2])
    
    for line in labdata:
        if line[2] in feature_timelines:
            feature_timelines[line[2]].append(line[4])
            feature_timelines[line[2]+"-times"].append(line[1])
        else:
            feature_timelines[line[2]]=list()
            feature_timelines[line[2]+"-times"]=list()
            feature_timelines[line[2]].append(line[4])
            feature_timelines[line[2]+"-times"].append(line[1]) 
            features.append(line[2])
            
    #With these feature timelines, assemble a dataframe of different features at different times
    #each row is a different time window.
    
    #The first step is to identify how many windows will be needed. We need the admittime and dischtime.
    admittime=demodata[demodata["patient"]==patient].admittime
    dischtime=demodata[demodata["patient"]==patient].dischtime
    diff=(dischtime-admittime).iloc[0]
    num_hours=diff.days*24+diff.seconds//3600
    print("Time of stay was ",num_hours," hours")
    
    #Calculate the number of windows, <=window_width hours = 1 window, window_width+1 hours = 2 windows, etc.
    if num_hours<=window_width:
        num_windows=1
    else:
        num_windows=num_hours-window_width+1
    print("Number of",window_width,"hour windows is",num_windows)
    
    #Initialize dataframe
    patient_data=X_ten[0:0]
    #Go through each window and add to feature vector
    for window in range(num_windows):
        a=a+1
        #print("Considering window",window)
        #For this window, define a begin time and an end time
        begin_time=admittime+timedelta(hours=window)
        end_time=begin_time+timedelta(hours=window_width)
        
        #Now it is time to loop through each of the features and filter them for events within this range
        for feature in features:
            #print("Considering feature",feature)
            feature_observations=list()
            for entrynum in range(len(feature_timelines[feature])):
                entry_time=feature_timelines[feature+"-times"][entrynum]
                if (entry_time <= end_time.iloc[0]) and (entry_time >= begin_time.iloc[0]):
                    feature_observations.append(feature_timelines[feature][entrynum])
            #If feature observations is still empty, just use average over whole series
            if len(feature_observations)==0 and len(feature_timelines[feature])>0:
                feature_observations.append(st.mean(feature_timelines[feature]))
                
            #Now calculate the statistics of each entry
            if(len(feature_observations)>1):
                mean=st.mean(feature_observations)
                median=st.median(feature_observations)
                stdev=st.stdev(feature_observations)
                mins=min(feature_observations)
                maxs=max(feature_observations)
            elif(len(feature_observations)==1):
                mean=st.mean(feature_observations)
                median=st.median(feature_observations)
                stdev=0
                mins=min(feature_observations)
                maxs=max(feature_observations)
            else: 
                mean="empty"
                median="empty"
                stdev="empty"
                mins="empty"
                maxs="empty"
            
            #Add patient feature data to patient data    
            patient_data.loc[a,feature+"-mean"]=mean
            patient_data.loc[a,feature+"-median"]=median
            patient_data.loc[a,feature+"-stdev"]=stdev
            patient_data.loc[a,feature+"-min"]=mins
            patient_data.loc[a,feature+"-max"]=maxs
    
    #Add missing data for each patient to these dataframes
    #Look through X_ten and find value, fill this in for all values in that column
    misc_features=list(patient_data.columns.values[-17:])
    for feature in misc_features:
        patient_data[feature]=X_ten[feature].iloc[0]
        
    #Add patient dataframe to list
    patient_dataframes[patient]=patient_data
    
pickle_jar=[patients,patient_dataframes]
pickle.dump(pickle_jar, open("patient-time-vectors.pkl",'wb') )
