#Combine numerical and one hot encoded data.
import pickle
import pandas as pd

#Change directory to where the data is
import os
os.chdir("../chartlabdata-any")

chartlabdata=pickle.load(open("clean-vectors.pkl",'rb'))[0]
categoricaldata=pickle.load(open("patient-categorical.pkl",'rb'))

patients=list(chartlabdata["patient"])

categorical_column_names=list(categoricaldata.columns)
small_categorical=pd.DataFrame(columns = categorical_column_names[0:])

#rename subject_id to patient
chartlabdata.rename(columns={'patient':'subject_id'},inplace=True)

for patient in patients:
    newrow=categoricaldata[categoricaldata["subject_id"]==patient].iloc[0,0:]
    small_categorical=small_categorical.append(newrow)

results=pd.merge(chartlabdata,
                 small_categorical,
                 on='subject_id',
                 sort=False)

pickle.dump(results, open("whole-dataset.pkl",'wb') )