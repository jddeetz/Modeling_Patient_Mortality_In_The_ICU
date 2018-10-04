# icu-risk-prediction: Predicting the mortality risk of patients in the intensive care unit

## Motivation/Problem
Physicians in hospital ICUs are overwhelmed with over 7,000 independent pieces of information each day, while simultaneously facing the pressure of making correct decisions, and long working hours. Emergency medical services are not well positioned to analyze all of the information at their disposal. A recent study estimates that physicians in the ICU make false negatives in predicting patient outcomes at a rate of 58% (Detsky, Harhay, and Bayard JAMA 2017).

The motivation of this project is to develop a model for patient mortality in the ICU based on commonly available information, such as vital signs and blood labwork. 

##Prerequisites
The code contained in this repository assumes that you have a working copy of the MIMIC-iii PostgreSQL database installed on your system. The pickle library is used for storing the outputs of several of the scripts between processing steps. Some standard machine learning libraries, such as numpy, pandas, sklearn, and tensorflow are also required.

## Description of Python Files

### QueryDB.py 
Used for the initial query to the Postgres database to derive information on a set of patients with specified ICD9 codes (physician diagnoses codes). Because the database query time on the chartevents table is quite long, due to the fact that it contains ~30GB of data, we only look at 100 patients here. 

### PreserveNumericalData.py
Preserves numerical information about the patients from their lab and chart events, discards textual and categorical information from these sources.

### CountLabChartItems.py
Counts the number of instances that each chart and lab item is listed for each patient.

### EliminateMissingItems.py
Eliminates chart and lab items that are not represented among at least 80% of the patients at least once. Also eliminates patients from the training data that are missing more than 20% of the features.

### NewQuery-Any.py
Performs a Database query on an expanded list of 2000 patients, but with a more limited set of chart and lab event items. 

### EasyTimeVectors.py
Extracts features from the chart and lab events time series data. Vectorizes this information by considering a 48 hour time window, over which numerical features are their statistics summarized.

### CleanData.py
Eliminates any patients with significant NaN's in their feature vectors. Replaces some patients NaN's with averages over other patients. 

### ExtractCategorical.py
Extracts patients ethnicity, gender, age, and other information from the postgres database

### CombineCategoricalNumerical.py
Combines categorical and numerical features for all patients

### LogReg.py
Performs logistic regression on full list of ~200 features and prints out summary statistics of the model.

### RandomForest.py
Performs random forest classification, and prints summary statistics of the model. Ranks feature importances and finds the top 25% most important features.

### NewLog.py
Performs logistic regression on a reduced set of ~50 features. Prints model statistics.

### NeuralNetwork.py
Uses a six layer artificial neural network to fit to the patient data on a reduced set of ~50 features. Prints out model statistics.