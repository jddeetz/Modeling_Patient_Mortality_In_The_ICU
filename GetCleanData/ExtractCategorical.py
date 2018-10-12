import psycopg2
import pickle
import pandas as pd
import functools

#Import postgresql database password
import sys
sys.path.insert(0, '../')
import pw
import os

#Change to working directory
os.chdir("../chartlabdata-any")

#Connect to mimic DB
conn = psycopg2.connect("dbname='mimic' user='postgres' host='localhost' password='"+password()+"'")

# calculate patient age at admission and store along with select columns from icustays table
icustays_query = """SELECT ie.subject_id, ie.hadm_id, ie.icustay_id,
ie.intime, ie.outtime, ie.first_careunit,
date_part('day', pat.dod_hosp - ie.intime) AS survival_days,
ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) AS age_admit
FROM mimiciii.icustays ie
INNER JOIN mimiciii.patients pat
ON ie.subject_id = pat.subject_id;
"""

# calculate patient age at death and store along with select columns from admissions table
admissions_query = """SELECT ad.subject_id, ad.hadm_id, ad.deathtime,
ad.insurance, ad.language, ad.religion, ad.marital_status, ad.admission_type,
case
    when ad.ethnicity like '%WHITE%' then 'WHITE'
    when ad.ethnicity like '%BLACK%' then 'BLACK'
    when ad.ethnicity like '%HISPANIC%' then 'HISPANIC'
    when ad.ethnicity like '%ASIAN%' then 'ASIAN'
    else 'OTHER' end as ethnicity,
ad.hospital_expire_flag,
ROUND((cast(ad.deathtime as date) - cast(pat.dob as date))/365.242, 2) AS age_death
FROM mimiciii.admissions ad
INNER JOIN mimiciii.patients pat
ON ad.subject_id = pat.subject_id
"""

# select columns from diagnoses_icd table
diagnoses_icd_query = """SELECT di.subject_id, di.hadm_id, di.icd9_code  
FROM mimiciii.diagnoses_icd di"""

# select columns from patients table
patients_query = """SELECT pat.subject_id, pat.gender FROM mimiciii.patients pat"""

# execute SQL queries and store in pandas dataframes
icustays_table = pd.read_sql_query(icustays_query,conn)
admissions_table = pd.read_sql_query(admissions_query,conn)
diagnoses_icd_table = pd.read_sql_query(diagnoses_icd_query,conn)
patients_table = pd.read_sql_query(patients_query,conn)
dfs = [icustays_table, admissions_table, diagnoses_icd_table, patients_table]


master = functools.reduce(lambda left,right: pd.merge(left,right,on='subject_id'), dfs)
master = master.drop_duplicates(subset=['subject_id', 'hadm_id'])

# get rid of unwanted columns
cols = ['hadm_id_y', 'hadm_id', 'intime', 'outtime', 'deathtime']
master.drop(cols, inplace=True, axis=1)

# create dummy variables for parameters and one-hot encode categorical variables
dummy_list = ['age_admit', 'gender', 'first_careunit', 'marital_status', 'ethnicity', 'admission_type', 'insurance']
dummy_frames = [pd.get_dummies(master[x]) for x in dummy_list]
dummy_frames = pd.concat(dummy_frames, axis=1)

#Concatenate one hot variables and other data
master3 = pd.concat([master, dummy_frames], axis=1)

#Specify training columns
train_cols = ['subject_id','age_admit', 
             'M',
             'CCU', 'CSRU',
             'DIVORCED', 'MARRIED', 'WIDOWED', 'SINGLE', 
             'ASIAN', 'BLACK', 'HISPANIC', 'WHITE', 
             'EMERGENCY', 'URGENT',
             'Medicaid', 'Medicare']

#Filter down the master data to the specified training columns
categoricals=master3[train_cols]

pickle_jar=categoricals
pickle.dump(pickle_jar, open("patient-categorical.pkl",'wb') )
