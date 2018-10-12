#Make predictions for patients
def predict(feature_vector):
    #Import some libraries
    import pickle
    import os
    import pandas as pd
    
    #Make prediction of mortality of patient
    pickle_jar=pickle.load(open("model.pkl",'rb'))
    model=pickle_jar[0]
    y_predict = model.predict_proba(feature_vector)
    
    params=model.coef_
    
    #Get top 5 reasons
    weights=pd.DataFrame(feature_vector*params)
    mean_weights=float(weights.mean(axis=1))
    stdev_weights=float(weights.std(axis=1))
    z_scores=(weights-mean_weights)/stdev_weights
    
    #for score_num in range(z_scores.shape[1]):
    sorted_z=pd.Series(z_scores.iloc[0,:]).sort_values(ascending=False)
    reason_list=pd.DataFrame(sorted_z.iloc[0:5]).T.columns.values
    
    return y_predict, reason_list
