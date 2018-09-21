import pandas as pd
import pickle

#Import library to run queries to postgresql database
import psycopg2

#Import postgresql database password
import sys
sys.path.insert(0, '/Users/jddeetz/Documents/Project/')
import pw

#Try to connect to database, and if can not connect, print a message
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
    cur.execute("""SELECT DIAGNOSES_ICD.subject_id, D_ICD_DIAGNOSES.long_title,DIAGNOSES_ICD.icd9_code
             FROM DIAGNOSES_ICD JOIN D_ICD_DIAGNOSES ON D_ICD_DIAGNOSES.icd9_code=DIAGNOSES_ICD.icd9_code 
             WHERE ascii(D_ICD_DIAGNOSES.icd9_code)<60 
             AND (CAST(coalesce(D_ICD_DIAGNOSES.icd9_code, '0') AS integer) = 584 
             OR CAST(coalesce(D_ICD_DIAGNOSES.icd9_code, '0') AS integer) = 5845 
             OR CAST(coalesce(D_ICD_DIAGNOSES.icd9_code, '0') AS integer) = 5846 
             OR CAST(coalesce(D_ICD_DIAGNOSES.icd9_code, '0') AS integer) = 5847 
             OR CAST(coalesce(D_ICD_DIAGNOSES.icd9_code, '0') AS integer) = 5848 
             OR CAST(coalesce(D_ICD_DIAGNOSES.icd9_code, '0') AS integer) = 5849
             )""")
    
    #Go through the query and grab the patient id's
    print("Retrieving patient list.")
    patients=[]
    rows=cur.fetchall()
    for row in rows:
        if(len(patients)>=500): continue #Only look at 10 patients for now
        if row[0] not in patients: #This account for multiple ICU visits by each patient.
            patients.append(row[0])
    
    #Let's get their admission and discharge times. Ethnicity and expire_flags
    admittime=[]
    dischtime=[]
    ethnicity=[]
    expire=[]
    print("Retrieving admission and discharge times, as well as ethnicity and mortalities.")
    for patient in patients:
        cur.execute("""SELECT admittime, dischtime, ethnicity, hospital_expire_flag 
                    FROM ADMISSIONS WHERE subject_id = """+str(patient))
        row=cur.fetchone()
        admittime.append(row[0])
        dischtime.append(row[1])
        ethnicity.append(row[2])
        expire.append(row[3])
    
    print("Retrieving age and sex of each patient.")
    dob=[]
    gender=[]
    for patient in patients:
            cur.execute("""SELECT dob, gender FROM PATIENTS WHERE subject_id = """+str(patient))
            row=cur.fetchone()
            dob.append(row[0])
            gender.append(row[1])
    
    #Calculate the age of each patient
    age=[]
    for patientnum in range(len(patients)):
        age.append(admittime[patientnum].year-dob[patientnum].year)
    agepd=pd.DataFrame(age,columns=["age"])        
    
    #Set up pandas dataframe to take in chart events data
    chartlabdata=pd.DataFrame(patients,columns=["patient"])
    print("Getting lab events of each patient.")
    a=-1
    for patient in patients:
        a=a+1
        cur.execute("""SELECT LABEVENTS.charttime, D_LABITEMS.label, LABEVENTS.value, LABEVENTS.valuenum, LABEVENTS.valueuom 
                    FROM LABEVENTS JOIN D_LABITEMS ON D_LABITEMS.itemid = LABEVENTS.itemid 
                    WHERE LABEVENTS.subject_id = """+str(patient))
        rows=cur.fetchall()
        for row in rows:
            if (isinstance(row[3],int) == True) or (isinstance(row[3],float) == True):
                if row[0].day<admittime[a].day+2:
                    if "Creatinine" == row[1]:
                        chartlabdata.loc[a,"Creatinine"]=row[3]
    
    #Construct Data Set
    chartlabdata["expire"]=expire
    chartlabdata["age"]=agepd
    chartlabdata["gender"]=gender

#    print("Getting chart events of each patient.")
#    items_label=[]
#    a=-1;
#    for patient in patients:
#        a=a+1
#        cur.execute("""SELECT CHARTEVENTS.charttime, D_ITEMS.label, CHARTEVENTS.value, CHARTEVENTS.valuenum, CHARTEVENTS.valueuom 
#                    FROM CHARTEVENTS JOIN D_ITEMS ON D_ITEMS.itemid = CHARTEVENTS.itemid 
#                    WHERE CHARTEVENTS.subject_id = """+str(patient))
#        rows=cur.fetchall()
#        for row in rows:
#            if (isinstance(row[3],int) == True) or (isinstance(row[3],float) == True):
#                if row[0].day<admittime[a].day+2:
#                    if "Creatinine" in row[1]:
#                        print(row)
#                        chartdata[row[1],]
    
    #Some summary statistics
    print("\n{:4.1f}".format(sum(expire)/len(expire)*100)," percent of patients died with these IC9 Codes")
        
    #Length of stay
    print("\nThe average length of stay:")
    print((pd.DataFrame(dischtime)-pd.DataFrame(admittime)).mean())
    
    print("\nThe median length of stay:")
    print((pd.DataFrame(dischtime)-pd.DataFrame(admittime)).median())
    
    print("\nThe standard deviation of the length of stay:")
    print((pd.DataFrame(dischtime)-pd.DataFrame(admittime)).std())
    
    print("\nThe maximum length of stay:")
    print((pd.DataFrame(dischtime)-pd.DataFrame(admittime)).max())
    
    print("\nThe minimum length of stay:")
    print((pd.DataFrame(dischtime)-pd.DataFrame(admittime)).min())
    
    #Different Ethnicities
    print("\nHere is a distribution of the ethnicities in the dataset:")
    print(pd.value_counts(pd.DataFrame(ethnicity,columns=['ethnicity'])["ethnicity"]))
    
    #Different Genders
    print("\nHere is a distribution of the genders in the dataset:")
    print(pd.value_counts(pd.DataFrame(gender,columns=['gender'])["gender"]))
    
    #Different Ages
    print("\nThe average age:")
    print(agepd.mean())
    
    print("\nThe median age:")
    print(agepd.median())
    
    print("\nThe standard deviation of the age:")
    print(agepd.std())
    
    print("\nThe maximum age:")
    print(agepd.max())
    
    print("\nThe minimum age:")
    print(agepd.min())
    
    pickle_jar=[chartlabdata]
    pickle.dump(pickle_jar, open("data.pkl",'wb') )
        
    
        
    
else: 
    print("There was no database connection and there is nothing left to do here.")