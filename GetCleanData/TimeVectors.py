#The goal of this code is to first generate time series vectors of each feature.
#For each item in chart and lab events, place them into a series

#Next, we would like to gather features from the last 48 hours of expired patients
#and the first 48 hours of nonexpired patients

#Define the width of the window
window_width = 48 

#Load some libraries
import pickle
import os
import numpy as np
import pandas as pd
import statistics as st
from datetime import datetime, timedelta

#Change directory to where files are being stored locally
os.chdir("../chartlabdata-any")

#Get hadm_id, expire, age, gender admittime, dischtime
demodata=pickle.load(open("data.pkl",'rb'))[0]

#First, let's get a list of the patients
picklejar = pickle.load(open("smallcounts.pkl",'rb'))  
chart_counts=picklejar[0]
lab_counts=picklejar[1]
patients=list(chart_counts["patients"])
    
a=0

for patient in patients:
    a=a+1
    #Open shortened chart and lab data files
    chartdata=pickle.load(open(str(patient)+"-smallerchart.pkl",'rb'))[0]
    labdata=pickle.load(open(str(patient)+"-smallerlab.pkl",'rb'))[0]
    

    #Create dictionary for feature timelines, each key will be an item of lab or chart events
    feature_timelines=dict()
    
    print("Unrolling chart and lab time series data for patient ",patient)
    #Unroll chart data
    first_hadm_id=int(demodata[demodata["patient"]==patient].hadm_id)
    
    #Assemble a list of the features
    features=list()
    
    #Unroll chart data
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
    
    #Unroll lab data
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
            
    if a==1:
        #Make a list of column headers for pandas
        column_header=[]
        column_header.append("patient")
        column_header.append("expire")
        for feature in features:
            column_header.append(feature+"-mean")
            column_header.append(feature+"-median")
            column_header.append(feature+"-stdev")
            column_header.append(feature+"-min")
            column_header.append(feature+"-max")
            Vectors=pd.DataFrame(columns=column_header)
    new_rows=pd.DataFrame(columns=column_header)    
        
    #Add some patient information    
    new_rows.loc[0,"patient"]=patient
    new_rows.loc[0,"expire"]=int(demodata[demodata["patient"]==patient].expire)
    
    admittime=demodata[demodata["patient"]==patient].admittime
    dischtime=demodata[demodata["patient"]==patient].dischtime
    
    #Now that the fatures have been unrolled into time,value series, We can start to assemble the vectors.
    if int(demodata[demodata["patient"]==patient].expire)==0:
        print("This patient did not expire")
        #begin_time=admittime+timedelta(hours=window_width)
        #end_time=admittime+timedelta(hours=window_width+24)
        begin_time=dischtime-timedelta(hours=window_width+24)
        end_time=dischtime-timedelta(hours=window_width)
        expire=0            
        
    else:
        print("This patient expired")
        begin_time=dischtime-timedelta(hours=window_width+24)
        end_time=dischtime-timedelta(hours=window_width)
        expire=1
        
    for feature in features:
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
        
        #Add to pandas dataframe
        new_rows.loc[0,feature+"-mean"]=mean
        new_rows.loc[0,feature+"-median"]=median
        new_rows.loc[0,feature+"-stdev"]=stdev
        new_rows.loc[0,feature+"-min"]=mins
        new_rows.loc[0,feature+"-max"]=maxs
        
    Vectors=pd.concat([Vectors, new_rows], ignore_index=True,sort=False)
#Pickle the vectors file
pickle_jar=[Vectors]
pickle.dump(pickle_jar, open("float-patient-vectors.pkl",'wb') )
                

        
