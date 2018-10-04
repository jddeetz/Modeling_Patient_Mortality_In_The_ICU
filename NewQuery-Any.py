#Based off our previous analysis of counting elements of the lab and chart data, we determined that
#only a small set of 47 features are actually present in great abundance, which are indicated below.

#Moreover, a lot of the features in labdata and chart data are actually redundant!

lab_features= ['Calcium, Total','Chloride','Creatinine','Glucose','Magnesium','Phosphate','Potassium',
               'Sodium','Urea Nitrogen','Hematocrit','Hemoglobin','MCH','MCHC','MCV','Platelet Count',
               'RDW','Red Blood Cells','White Blood Cells','Alanine Aminotransferase (ALT)','Albumin',
               'Anion Gap','Asparate Aminotransferase (AST)','Bicarbonate','pH','Specific Gravity',
               'Basophils','Eosinophils','Lymphocytes','Monocytes','Neutrophils','INR(PT)','PT','PTT']

chart_features= ['Braden Score','Eye Opening','GCS Total','Heart Rate','Motor Response',
                 'NBP [Systolic]','NBP Mean','Respiratory Rate','SpO2','Temperature C (calc)',
                 'Temperature F','Verbal Response','Carbon Dioxide','NBP [Diastolic]']

#We should find some patients with specified ICD9 codes that died of that condition.
#When expire_flag=1

#Specify how many expired patients you would like
num_expired=1000

#Import library to run queries to postgresql database
import psycopg2

#Import Pickle
import pickle

#Import pandas
import pandas as pd

#Import postgresql database password
import sys
sys.path.insert(0, '../')
import pw
import os

os.chdir("../chartlabdata-any")

try:
    conn = psycopg2.connect("dbname='mimic' user='postgres' host='localhost' password='"+password()+"'")
except:
    print("I am unable to connect to the database")
    
#If the database connection was successful, we are ready to run some queries on it.
if("conn" in locals()):
    print("Connection to database successful")
    
    #Define a cursor to work with
    cur = conn.cursor()
 
    #Now that we have the cursor defined we can set the default search path of the cursor to mimiciii.
    cur.execute("""SET search_path TO mimiciii;""")
 
    #Run a query to get the ICD9 codes of patients with acute kidney injuries.
    cur.execute("""SELECT DIAGNOSES_ICD.subject_id, D_ICD_DIAGNOSES.long_title,DIAGNOSES_ICD.icd9_code,DIAGNOSES_ICD.hadm_id
             FROM DIAGNOSES_ICD JOIN D_ICD_DIAGNOSES ON D_ICD_DIAGNOSES.icd9_code=DIAGNOSES_ICD.icd9_code 
             WHERE ascii(D_ICD_DIAGNOSES.icd9_code)<60 LIMIT 200000""")
    
    #Go through the query and grab the patient id's
    print("Retrieving patient list and corresponding hadm_id.")
    patients=[]
    hadm_id=[]
    rows=cur.fetchall()
    for row in rows:
        if row[0] not in patients: #This account for multiple ICU visits by each patient.
            patients.append(row[0])
            hadm_id.append(row[3])
            
    print("Found ",len(patients)," patients with one of these ICD9 codes")
 
    #Let's get their admission and discharge times. Ethnicity and expire_flags
    admittime=[]
    dischtime=[]
    ethnicity=[]
    expire=[]
    new_patients=[]
    new_hadm=[]
    
    #Find the expire flags of these patients.
    expired_patients=0
    living_patients=0
    for patientnum in range(len(patients)):
        patient=patients[patientnum]
        hadm=hadm_id[patientnum]
        if expired_patients<num_expired:
            cur.execute("""SELECT hadm_id, admittime, dischtime, ethnicity, hospital_expire_flag 
                        FROM ADMISSIONS WHERE hospital_expire_flag = 1 AND subject_id = """+str(patient)+""" AND hadm_id = """+str(hadm))
            try: 
                row=cur.fetchone()
                new_hadm.append(row[0])
                admittime.append(row[1])
                dischtime.append(row[2])
                ethnicity.append(row[3])
                expire.append(row[4])
                new_patients.append(patient)
                expired_patients=expired_patients+1
            except:
                print("Skipping patient ",patient)
        
        elif living_patients<num_expired: #Get an equal number of living patients
            cur.execute("""SELECT hadm_id, admittime, dischtime, ethnicity, hospital_expire_flag 
            FROM ADMISSIONS WHERE hospital_expire_flag = 0 AND subject_id = """+str(patient)+""" AND hadm_id = """+str(hadm))
            
            try: 
                row=cur.fetchone()
                new_hadm.append(row[0])
                admittime.append(row[1])
                dischtime.append(row[2])
                ethnicity.append(row[3])
                expire.append(row[4])
                new_patients.append(patient)
                living_patients=living_patients+1
            except:
                print("Skipping patient ",patient)
    
    print("Retrieving age and sex of each patient.")
    patients=new_patients
    hadm_id=new_hadm
    dob=[]
    gender=[]
    for patient in patients:
            cur.execute("""SELECT dob, gender FROM PATIENTS WHERE subject_id = """+str(patient))
            row=cur.fetchone()
            dob.append(row[0])
            gender.append(row[1])
           
    #Ok, now we have the categorical and index information of an equal number of expired and living patients. Great!
    
    #Next, we should put some numbers to chart and lab items listed above.
    lab_item_codes=[]
    chart_item_codes=[]
    
    for feature in lab_features:
        cur.execute("""SELECT itemid FROM D_LABITEMS WHERE label = '"""+feature+"""'""")
        rows=cur.fetchall()
        for row in rows:
            lab_item_codes.append(row[0])
        
    for feature in chart_features:
        cur.execute("""SELECT itemid FROM D_ITEMS WHERE label = '"""+feature+"""'""")
        rows=cur.fetchall()
        for row in rows:
            chart_item_codes.append(row[0])
    
    #Some tests are redundant, i.e. there is more than one way to measure heart rate, blood glucose etc.
    
    #Ok, now that we have the ids of the chart and lab items, we should determine for chart event items 
    #which partitions of chart events table each of these is actually inside of.
    
    part_for_chart_items=[]
    for code in chart_item_codes:
        for partition in range(1,207): #Loop through partitions 1 to 207
            cur.execute("""SELECT itemid FROM CHARTEVENTS_"""+str(partition)+""" WHERE itemid = """+str(code))
            rows=cur.fetchall()
            if len(rows)>0:
                part_for_chart_items.append(partition)
                print("Found partition number for ",code)
    
    #Query lab events
    for patientnum in range(len(patients)):
        patient=patients[patientnum]
        hadm=hadm_id[patientnum]
        cur.execute("""SELECT LABEVENTS.hadm_id, LABEVENTS.charttime, D_LABITEMS.label, LABEVENTS.value, LABEVENTS.valuenum, LABEVENTS.valueuom 
                    FROM LABEVENTS JOIN D_LABITEMS ON D_LABITEMS.itemid = LABEVENTS.itemid 
                    WHERE LABEVENTS.subject_id = """+str(patient)+""" AND D_LABITEMS.itemid IN """+str(tuple(lab_item_codes))+""" AND LABEVENTS.hadm_id = """+str(hadm))
        pickle_jar=cur.fetchall()
        print("Saving lab events data for patient number ",str(patient))
        pickle.dump(pickle_jar, open(str(patient)+"-labdata.pkl",'wb') )
    
    #Ok, now that we have the partition number for each of these codes. It is time to query the chart events partitions
    #Query chart events
    chartinfo=dict()
    for item_num in range(len(chart_item_codes)):
        print("Looking at item number ",item_num+1," of ",len(chart_item_codes))
        part=part_for_chart_items[item_num]
        item=chart_item_codes[item_num]
        cur.execute("""SELECT CHARTEVENTS_"""+str(part)+""".subject_id, CHARTEVENTS_"""+str(part)+""".hadm_id, CHARTEVENTS_"""+str(part)+""".charttime, D_ITEMS.label, CHARTEVENTS_"""+str(part)+""".value, 
                    CHARTEVENTS_"""+str(part)+""".valuenum, CHARTEVENTS_"""+str(part)+""".valueuom FROM CHARTEVENTS_"""+str(part)+""" 
                    JOIN D_ITEMS ON D_ITEMS.itemid = CHARTEVENTS_"""+str(part)+""".itemid 
                    WHERE CHARTEVENTS_"""+str(part)+""".itemid = """+str(item)+""" AND CHARTEVENTS_"""+str(part)+""".subject_id IN """+str(tuple(patients))+""" AND CHARTEVENTS_"""+str(part)+""".hadm_id IN """+str(tuple(hadm_id)))
        rows=cur.fetchall()
        
        #Now we need to parse this info for each patient with a dict of lists
        for row in rows:
            if row[0] not in chartinfo:
                chartinfo[row[0]]=list()
                chartinfo[row[0]].append(row[1:])
            else:
                chartinfo[row[0]].append(row[1:])
        
        
    for patientkey in chartinfo:
        pickle_jar=chartinfo[patientkey]
        print("Saving chart events data for patient number ",str(patientkey))
        pickle.dump(pickle_jar, open(str(patientkey)+"-chartdata.pkl",'wb'))
    
    #Calculate the age of each patient
    age=[]
    for patientnum in range(len(patients)):
        age.append(admittime[patientnum].year-dob[patientnum].year)
    agepd=pd.DataFrame(age,columns=["age"])  
    
    #Save some categorical data
    chartlabdata=pd.DataFrame(patients,columns=["patient"])
    chartlabdata["hadm_id"]=hadm_id
    chartlabdata["expire"]=expire
    chartlabdata["age"]=agepd
    chartlabdata["gender"]=gender
    chartlabdata["admittime"]=admittime
    chartlabdata["dischtime"]=dischtime
    pickle_jar=[chartlabdata]
    pickle.dump(pickle_jar, open("data.pkl",'wb') )
        
else: 
    print("There was no database connection and there is nothing left to do here.")       
        