#The purpose of this code is to open pkl files containing lab and chart data for each patient.
#We want to remove any data from these files that is not numerical. 

#Import some packages
import pickle
import os
import pandas as pd
from shutil import copyfile


#Change directory to where files are being stored locally
os.chdir("../chartlabdata-any")

#Load data.pkl, the file containing the patient numbers
picklejar = pickle.load(open("data.pkl",'rb'))  
patients=list(picklejar[0]["patient"])

#Now that we have a list of the patients, we can loop through each of the data files
#and remove any entries that contain strings.
for patient in patients:
    
    #Open the pkl file corresponding to chart-data for this patient
    print("Opening chart data file for patient ",patient)
    try:
        chartdata=pickle.load(open(str(patient)+"-chartdata.pkl",'rb'))
    except:
        copyfile("9999-chartdata.pkl", str(patient)+"-chartdata.pkl")
        copyfile("9999-labdata.pkl", str(patient)+"-labdata.pkl")
        chartdata=pickle.load(open(str(patient)+"-chartdata.pkl",'rb'))
    
    #From the chart data, create a smaller file only containing the numerical chart data
    smallchart=list()
    #This is what a string of chartdata looks like
    #(145576, datetime.datetime(2195, 4, 6, 8, 0), 'CVP', '11', 11.0, 'mmHg')
    #We want to test to see if the fifth column contains a floating point number.
    for line in chartdata:
        if line[4] != None:
            smallchart.append(line)
            
    pickle_jar=[smallchart]
    pickle.dump(pickle_jar, open(str(patient)+"-smallchart.pkl",'wb') )
    
    #Open the pkl file corresponding to lab-data for this patient
    print("Opening lab data file for patient ",patient)
    labdata=pickle.load(open(str(patient)+"-labdata.pkl",'rb'))
    
    #From the lab data, create a smaller file only containing the numerical lab data
    smalllab=list()
    #This is what a string of labdata looks like
    #(199309, datetime.datetime(2115, 4, 18, 4, 49), 'PT', '18.5', 18.5, 'sec')
    #We want to test to see if the fifth column contains a floating point number.
    for line in labdata:
        if line[4] != None:
            smalllab.append(line)
            
    pickle_jar=[smalllab]
    pickle.dump(pickle_jar, open(str(patient)+"-smalllab.pkl",'wb') )
