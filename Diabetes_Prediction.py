#!/usr/bin/env python
# coding: utf-8

# This notebook is used to make predictions of diabetes based on known features. The purpose of this exercise is to demonstrate the power of Machine learning. The dataset contains a range of health-related attributes collected to aid the development of predictive models to identify any risk of diabetes. I will build the model and indicate each phase that was undergoing. Ultimately, I will write detailed documentation explaining everything that happened and my findings. The inspiration for this work is due to my passion for working in the healthcare field and my data science knowledge.
# Below are details of this data and columns, along with a link to the data for more information.
# 
# Columns:
# 1.	Id: Unique identifier for each data entry.
# 2.	Pregnancies: Number of times pregnant.
# 3.	Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
# 4.	BloodPressure: Diastolic blood pressure (mm Hg).
# 5.	SkinThickness: Triceps skinfold thickness (mm).
# 6.	Insulin: 2-Hour serum insulin (mu U/ml).
# 7.	BMI: Body mass index (weight in kg / height in m^2).
# 8.	DiabetesPedigreeFunction: Diabetes pedigree function, a genetic score of diabetes.
# 9.	Age: Age in years.
# 10.	Outcome: Binary classification indicating the presence (1) or absence (0) of diabetes.
# 
# Link to data: https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes

# ## Import needed libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle


# ## Load the data

diabetes_df = pd.read_csv('Healthcare-Diabetes.csv')
diabetes_df.head()



# Get the shape of data
diabetes_df.shape


# Cheak information about the data
diabetes_df.info()



# Confirm that there is no missing values
diabetes_df.isnull().sum()



# Count the number distinct element
diabetes_df['Id'].nunique()



# Check for statistical propertices of the data
diabetes_df.describe()


# Count the element in the outcome 
diabetes_df['Outcome'].value_counts()


# Check for unique Age
diabetes_df['Age'].unique()


# Get the count of each Age int he dataset
diabetes_df['Age'].nunique()


# Create age buckets
age_buckets = ["[{0} - {1})".format(age_range, age_range + 10) for age_range in range(20, 100, 10)]
diabetes_df['age_range'] = pd.cut(diabetes_df['Age'], bins=8, labels=age_buckets)


diabetes_df.head()



# Check the Age range count
diabetes_df['age_range'].value_counts()


# Drop the Age and ID column
diabetes_df.drop(['Id', 'Age'], axis=1, inplace=True)


diabetes_df


# ## Data Validation

# Create a contingency table between the age range and Outcome
crosstab_01 = pd.crosstab(diabetes_df['age_range'], diabetes_df['Outcome'])
crosstab_01


# Plot 
crosstab_01.plot(kind='bar', stacked = True)


crosstab_norm = crosstab_01.div(crosstab_01.sum(1), axis = 0)
crosstab_norm



crosstab_norm.plot(kind='bar', stacked = True,
                title = 'Bar Graph of Age range with Response Overlay')

diabetes_df.dtypes


# Label Encode the age range

# Initialize the Label Encoder
age_encode = LabelEncoder()
diabetes_df['age_range_encoded'] = age_encode.fit_transform(diabetes_df['age_range'])


diabetes_df


# Define the feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'age_range_encoded']


# ## Data Scaling

# Scale the data
scaler = MinMaxScaler()
scaler.fit(diabetes_df[feature_names])



# ## Preparing data for the model

X = diabetes_df[feature_names].values
y = diabetes_df['Outcome'].values


# ## Train | Test Split


from sklearn.model_selection import train_test_split


# help(train_test_split)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))



# Import the models
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay



# Instantiate the models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='scale')))
models.append(('RFC', RandomForestClassifier(n_estimators=100)))
models.append(('DTR', DecisionTreeRegressor()))



# Defind and empty list to hold the name and results

results = []
names = []



# Create a function to capture the various models
for name, model in models:
    
    
    # Creating K-Fold Cross-Validator
    kfold = model_selection.KFold(n_splits=10)
    
     # Cross-Validation
    cross_val_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
                                                        
    # Storing Results
    results.append(cross_val_results)
    names.append(name)
                                                        
    # Printing Results
    message = "{}: {} ({})".format(name, cross_val_results.mean(), cross_val_results.std())
    print(message)


# Initialize the Random Forest Classifier
RandomForestClassifier_model = RandomForestClassifier(n_estimators=100)

# Train the model
RandomForestClassifier_model.fit(X_train, y_train)

# Make predictions
RandomForestClassifier_model_prediction = RandomForestClassifier_model.predict(X_test)

# Calculate accuracy
RandomForestClassifier_model_accuracy = accuracy_score(y_test, RandomForestClassifier_model_prediction)
print(f"Accuracy: {RandomForestClassifier_model_accuracy:.2f}")


# ## Evaluate the Trained Model
predictions = RandomForestClassifier_model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels   : ' ,y_test[:10])



# Creating the classification report
print('Random Forest Classifier: \n', classification_report(y_test, predictions, target_names=['0', '1']))



# Creating a confusion matix
con_RandomForestClassifier_matrix = confusion_matrix(y_test, predictions)
con_RandomForestClassifier_matrix_display = ConfusionMatrixDisplay(con_RandomForestClassifier_matrix) 
fig, ax = plt.subplots(figsize=(10,10))
con_RandomForestClassifier_matrix_display.plot(cmap=plt.cm.Blues, ax=ax)


# ## Save the Model


rfc_pickle = open('random_forest_classifier_model.pkl', 'wb')
pickle.dump(RandomForestClassifier_model,rfc_pickle)
rfc_pickle.close()

#print("Model loaded successfully!")


# Assuming you have new data in a DataFrame `new_data`
'''
new_data = np.array([[6,148,72,35,0,33.6,0.627,50]]).astype('float64')
print('New Data: {}'.format(list(new_data[0])))
5
# Predict on the new data
predictions = model.predict(new_data)

# Print the predictions
print("Predictions on new data:", predictions)
'''







