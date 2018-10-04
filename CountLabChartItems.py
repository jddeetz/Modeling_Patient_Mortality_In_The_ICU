#The purpose of this code is to determine if certain items in the lab events or
#chart events are adequately represented across all patients to be included for
#a classification model.

#Load some libraries
import pickle
import os
import pandas as pd

#Change directory to where files are being stored locally
os.chdir("../chartlabdata-any")

#Load data.pkl, the file containing the patient numbers
picklejar = pickle.load(open("data.pkl",'rb'))  
patients=list(picklejar[0]["patient"])

#Now that we have a list of the patients, we can loop through each of the patients
#lab and chart files and count the number of instances each item occurs.

chart_counts=pd.DataFrame()
chart_counts["patients"]=patients

lab_counts=pd.DataFrame()
lab_counts["patients"]=patients

a=-1
for patient in patients:
    a=a+1
    #Open the pkl file corresponding to chart-data for this patient
    print("Opening chart data file for patient ",patient)
    smallchart=pickle.load(open(str(patient)+"-smallchart.pkl",'rb'))
    
    for line in smallchart[0]:
        #This is what a string of chartdata looks like
        #(145576, datetime.datetime(2195, 4, 6, 8, 0), 'CVP', '11', 11.0, 'mmHg')
        try:
            chart_counts[line[2]][a]=chart_counts[line[2]][a]+1
        except:
            chart_counts[line[2]]=0
            chart_counts[line[2]][a]=chart_counts[line[2]][a]+1
    
    #Open the pkl file corresponding to lab-data for this patient
    print("Opening lab data file for patient ",patient)
    smalllab=pickle.load(open(str(patient)+"-smalllab.pkl",'rb'))
    
    for line in smalllab[0]:
        #This is what a string of labdata looks like
        #(199309, datetime.datetime(2115, 4, 18, 4, 49), 'PT', '18.5', 18.5, 'sec')
        try:
            lab_counts[line[2]][a]=lab_counts[line[2]][a]+1
        except:
            lab_counts[line[2]]=0
            lab_counts[line[2]][a]=lab_counts[line[2]][a]+1
            
#Ok, now we have the counts of each type of item in chart and lab events
#Let's save this information.
            
pickle_jar=[chart_counts,lab_counts]
pickle.dump(pickle_jar, open("itemcounts.pkl",'wb') )
