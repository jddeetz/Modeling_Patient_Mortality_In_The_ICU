#The purpose of this code is to eliminate items and patients missing significant amounts of entries 
#from lab and chart data

#If an item is missing for more than a specified fraction of the patients, it might not be worth
#including in the dataset. This cutoff can be specified here:

fraction_missing_tolerance=0.20

#Load some libraries
import pickle
import os
import pandas as pd

#Change directory to where files are being stored locally
os.chdir("../chartlabdata-any")

#Load itemcounts.pkl, the file containing the counts of items for each patient 
#reported in their chart and lab data
picklejar = pickle.load(open("itemcounts.pkl",'rb'))  
chart_counts=picklejar[0]
lab_counts=picklejar[1]

print("\n\nDetermining if chart items have sufficient data")
#Get a list of the column items in chart data
chart_items=list(chart_counts.columns.values)

chart_kill_list=list()
for item in chart_items:
    number_present=sum((chart_counts[item]!=0)*1)
    fraction_present=number_present/chart_counts.shape[0]    
    if fraction_present<1-fraction_missing_tolerance:
        chart_kill_list.append(item)
    else: print(item," has sufficient data")

print("\n\nDetermining if lab items have sufficient data")
#Get a list of the column items in lab data
lab_items=list(lab_counts.columns.values)

lab_kill_list=list()
for item in lab_items:
    number_present=sum((lab_counts[item]!=0)*1)
    fraction_present=number_present/lab_counts.shape[0]    
    if fraction_present<1-fraction_missing_tolerance:
        lab_kill_list.append(item)
    else: print(item," has sufficient data")
    
#Go through each patients files and remove unecessary items
    
#Get patient id numbers
picklejar = pickle.load(open("data.pkl",'rb'))  
patients=list(picklejar[0]["patient"])

#Remove items from labcounts and chartcounts dataframes
for item in chart_kill_list:
    del chart_counts[item]

for item in lab_kill_list:
    del lab_counts[item]

#If a patient is missing a significant fraction of the items, it might not be worth
#including in the dataset. 
    
patient_ignore_list=list()
for patientnum in range(chart_counts.shape[0]):
    number_present=sum((chart_counts.iloc[patientnum,:]!=0)*1)
    fraction_present=number_present/chart_counts.shape[1]
    if fraction_present<1-fraction_missing_tolerance:
        patient_ignore_list.append(patientnum)
        print(patientnum," does not have sufficient chart data")
    
    number_present=sum((lab_counts.iloc[patientnum,:]!=0)*1)
    fraction_present=number_present/lab_counts.shape[1]
    if fraction_present<1-fraction_missing_tolerance:
        patient_ignore_list.append(patientnum)
        print(patientnum," does not have sufficient lab data")

#Remove patient numbers from dataframes
#Convert patient num into patient id
patientid_ignore_list=chart_counts.patients[patient_ignore_list]
#Make a list of unique patient ids
patientid_ignore_list=list(set(patientid_ignore_list))

for patientid in patientid_ignore_list:
    chart_counts=chart_counts[chart_counts.patients!=patientid]
    lab_counts=lab_counts[lab_counts.patients!=patientid]

#Save chart and lab counts
pickle_jar=[chart_counts,lab_counts]
pickle.dump(pickle_jar, open("smallcounts.pkl",'wb') )

#Loop through files
for patient in patients:
    #Do not save patient files for those in the ignore list
    if patient in patientid_ignore_list: continue 
    
    print("Shrinking chart and lab data files for patient",patient)
    
    #Open the pkl file corresponding to chart-data for this patient
    smallchart=pickle.load(open(str(patient)+"-smallchart.pkl",'rb'))
    
    smallerchart=list()
    for line in smallchart[0]:
        #This is what a string of chartdata looks like
        #(145576, datetime.datetime(2195, 4, 6, 8, 0), 'CVP', '11', 11.0, 'mmHg')
        if line[2] not in chart_kill_list: smallerchart.append(line)
        
    pickle_jar=[smallerchart]
    pickle.dump(pickle_jar, open(str(patient)+"-smallerchart.pkl",'wb') )
    
    #Open the pkl file corresponding to lab-data for this patient
    smalllab=pickle.load(open(str(patient)+"-smalllab.pkl",'rb'))
    
    smallerlab=list()
    for line in smalllab[0]:
        if line[2] not in lab_kill_list: smallerlab.append(line)
        
    pickle_jar=[smallerlab]
    pickle.dump(pickle_jar, open(str(patient)+"-smallerlab.pkl",'wb') )
    
#Record unique features
chartdata_columns=list(chart_counts.columns)
labdata_columns=list(lab_counts.columns)



