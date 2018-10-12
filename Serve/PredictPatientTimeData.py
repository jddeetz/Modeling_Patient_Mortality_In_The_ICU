from MakePredictions import predict
import pandas as pd
import pickle
import os

os.chdir("../chartlabdata-any")

pickle_jar=pickle.load(open("patient-time-vectors.pkl",'rb'))

patients=pickle_jar[0]
patient_time_windows=pickle_jar[1]

for patient in patients:
    print("Patient number : ",patientnum)
    patient_time_window=patient_time_windows[patient]
        
    for window_num in range(patient_time_window.shape[0]):
        features=pd.DataFrame(patient_time_window.iloc[window_num,:]).T
        (y_predict , reason_list) = predict(features)
        print(y_predict[0][1])
        if y_predict[0][1] > 0.5:#print reason list above this threshold.
            print(reason_list)
        
        



