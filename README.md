# Modeling_Patient_Mortality_In_The_ICU: Predicting the mortality risk of patients in the intensive care unit

## Motivation/Problem
Physicians in hospital ICUs are overwhelmed with over 7,000 independent pieces of information each day, while simultaneously facing the pressure of making correct decisions, and long working hours. Emergency medical services are not well positioned to analyze all of the information at their disposal. A recent study estimates that physicians in the ICU make false negatives in predicting patient outcomes at a rate of 58% (Detsky, Harhay, and Bayard JAMA 2017).

The motivation of this project is to develop a model for patient mortality in the ICU based on commonly available information, such as vital signs and blood labwork. 

A link to the powerpoint presentation detailing the results of this project: https://tinyurl.com/ybdufeau

A link to a dashboard depicting patient mortalities and risk factors: http://vulcanapp.xyz

A image of the dashboard is below, which displays the mortality risk of each patient and corresponding risk factors.

![alt text](https://i.imgur.com/a7weOAk.png)

## Prerequisites
The code contained in this repository assumes that you have a working copy of the MIMIC-iii PostgreSQL database installed on your system. The pickle library is used for storing the outputs of several of the scripts between processing steps. Some standard machine learning libraries, such as numpy, pandas, sklearn, and tensorflow are also required.

## How to Use
### How to Form Training Set Data
In order to form the training set data, a copy of the MIMIC-iii Postres database must be installed. "QueryDB.py" will extract the necessary data, whereas "PreserveNumericalData.py", "CountLabChartItems.py", "EliminateMissingItems.py", "EasyTimeVectors.py", "CleanData.py" will formulate numerical data from patient lab and chart items. Categorical data about each patient is extracted using "ExtractCategorical.py". The categorical and numerical patient data are combines using "CombineNumericalCategorical.py". A more detailed description of each of these are below. These files are located in the GetCleanData directory.

### How to Train Models 
Models can be trained using LogReg.py or RandomForest.py for these respective models. RandomForest.py also performs feature selection and needs to be run before NewLogReg.py or NeuralNetwork.py can be run, as these operate on a reduced feature set. These scripts are in the TrainModel directory. A sample dataset to get users up and running, without going through the messy data cleaning process outlined above, has been placed in the chartlabdata-any directory.

### How to Make Inferences Using the Model
LogReg.py and NewLogReg.py generate pickle files containing logistic regression objects that can be called on for inference. As examples, these pickle files also contain feature vectors for ten patients. Some python files for making inferences on the model are located in the Serve directory.

## Description of Python Files

### QueryDB.py 
Performs a Database query on a sample of 2000 patients, but with a limited set of chart and lab event items that are represented by a substantial number of the patients. Used for the initial query to the Postgres database to derive information on a set of patients.

### PreserveNumericalData.py
Preserves numerical information about the patients from their lab and chart events, discards textual and categorical information from these sources.

### CountLabChartItems.py
Counts the number of instances that each chart and lab item is listed for each patient.

### EliminateMissingItems.py
Eliminates chart and lab items that are not represented among at least 80% of the patients at least once. Also eliminates patients from the training data that are missing more than 20% of the features.

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
