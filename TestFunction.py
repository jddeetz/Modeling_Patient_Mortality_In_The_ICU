from MakePredictions import predict
import pandas as pd
import pickle
import os

os.chdir("../chartlabdata-any")

pickle_jar=pickle.load(open("model.pkl",'rb'))

X_ten=pickle_jar[1]
y_ten=pickle_jar[2]
patients=pickle_jar[3]

for patientnum in range(len(patients)):
    print("Patient number : ",patientnum)
    patient_vector=pd.DataFrame(X_ten.iloc[patientnum,:]).T
    (y_predict , reason_list) = predict(patient_vector)
    print(y_predict[0][1])
    if y_predict[0][1] > 0.5:#print reason list above this threshold.
        print(reason_list)
    
    