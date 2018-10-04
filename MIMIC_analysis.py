
# coding: utf-8

# ## Import relevant modules

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mt
import seaborn as sns
import numpy as np
import datetime as dt
import math
import psycopg2
import statsmodels.api as sm
import brewer2mpl
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from flask import Flask
from flask import Flask, render_template


# ## Query the MIMIC database for relevant patient information

# In[4]:


conn = psycopg2.connect(dbname="mimic", user="pradeepbandaru", password="RasRafB2H")

# calculate patient age at admission and store along with select columns from icustays table
icustays_query = """SELECT ie.subject_id, ie.hadm_id, ie.icustay_id,
ie.intime, ie.outtime, ie.first_careunit,
date_part('day', pat.dod_hosp - ie.intime) AS survival_days,
ROUND((cast(ie.intime as date) - cast(pat.dob as date))/365.242, 2) AS age_admit
FROM mimiciii.icustays ie
INNER JOIN mimiciii.patients pat
ON ie.subject_id = pat.subject_id;
"""
# WHERE pat.expireflag = 1

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


# In[5]:


# combine all tables into one master table per patient
dfs = [icustays_table, admissions_table, diagnoses_icd_table, patients_table]
master = reduce(lambda left,right: pd.merge(left,right,on='subject_id'), dfs)
master = master.drop_duplicates(subset=['subject_id', 'hadm_id'])

# get rid of unwanted columns
cols = ['hadm_id_y', 'hadm_id', 'intime', 'outtime', 'deathtime']
master.drop(cols, inplace=True, axis=1)


# ## Filter master patient table by disease-specific ICD9 codes

# In[6]:


#master.head(15)
#print df_final.shape

# filter the master table by patients who have died in the hospital
#master2 = master.query('age_admit > 100')
master2 = master[master.age_admit > 100]

#filter the master table by icd9 codes that correspond to AKI or MI
#MI:
master2 = master.loc[master['icd9_code'].isin(['41011', '41091', '412', '4139', '41401', '41400', '41000', '41010'])]

#AKI:
#master2 = master.loc[master['icd9_code'].isin(['5939', '5845', '5846', '5847', '5848', '5849'])]


# ## Train a regression model

# In[7]:


# remove missing data
#master2 = master2.dropna(axis=0)

# create dummy variables for parameters and one-hot encode categorical variables
dummy_list = ['age_admit', 'gender', 'first_careunit', 'marital_status', 'ethnicity', 'admission_type', 'insurance']
dummy_frames = [pd.get_dummies(master2[x]) for x in dummy_list]
dummy_frames = pd.concat(dummy_frames, axis=1)
#master2.shape


# In[8]:


master2['year_die'] = (master2.survival_days < 365).astype(int)
master3 = pd.concat([master2, dummy_frames], axis=1)
master3['intercept'] = 1.0
# train_cols = ['intercept','age_admit', 'CSRU', 'MICU', 'SICU', 'CCU', 
#              'DIVORCED', 'MARRIED', 'SEPARATED', 'WIDOWED', 'SINGLE', 
#              'ASIAN', 'BLACK', 'HISPANIC', 'WHITE', 
#              'EMERGENCY', 'URGENT',
#              'Medicaid', 'Medicare', 'Private', 'Government']


# MI features
train_cols = ['intercept','age_admit', 
             'M',
             'CCU', 'CSRU',
             'DIVORCED', 'MARRIED', 'WIDOWED', 'SINGLE', 
             'ASIAN', 'BLACK', 'HISPANIC', 'WHITE', 
             'EMERGENCY', 'URGENT',
             'Medicaid', 'Medicare']


# AKI features
# train_cols = ['intercept','age_admit', 
#              'MICU', 'SICU', 'CCU', 
#              'DIVORCED', 'MARRIED', 'WIDOWED', 'SINGLE', 
#              'ASIAN', 'BLACK', 'HISPANIC', 'WHITE', 
#              'EMERGENCY', 'URGENT',
#              'Medicaid', 'Medicare']


# In[19]:


logit = sm.Logit(master3.year_die, master3[train_cols])
result = logit.fit()
result.summary()


# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(master3[train_cols], master3.year_die, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[22]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print len(y_pred)
print y_pred.tolist()
print y_test.tolist()

X_test['prediction'] = y_test
display_table = X_test.to_dict('records')
print display_table


# In[18]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('dashboard.html', display_table = display_table)
if __name__ == '__main__':
    app.run(debug=True)