
# Python for Data Analysis Part 30: Random Forests
# http://hamelg.blogspot.com/2015/12/python-for-data-analysis-part-30-random.html?view=classic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from enum import Enum

class FeatureColumn(Enum):
    Sex = 0
    Pclass = 1
    SibSp = 2
    Embarked = 3
    Age = 4
    Fare = 5

def main():
    
#     1. TRAIN THE MODEL
#     get training data 
    df_titanic_train = pd.read_csv("train.csv")
  
#     data preprocessing - fill nan
    df_titanic_train = df_titanic_train.fillna(df_titanic_train.median()[:FeatureColumn.Age.value], inplace=True)        
    df_titanic_train["Embarked"] = df_titanic_train["Embarked"].fillna("S")      
    df_titanic_train["Fare"] = df_titanic_train["Fare"].fillna(0)     
    
#     data preprocessing - convert string features to numeric values
    label_encoder = preprocessing.LabelEncoder()
    
#     print label mapping
#     label_encode_sex = label_encoder.fit(df_titanic_train["Sex"])    
#     for i, item in enumerate(label_encode_sex.classes_):
#         print(item, '-->', i)
    
#     male = 1, female = 0
    df_titanic_train["Sex"] = label_encoder.fit_transform(df_titanic_train["Sex"])
        
#     C = 0, Q = 1, S = 2
    df_titanic_train["Embarked"] = label_encoder.fit_transform(df_titanic_train["Embarked"])
    
#     set the random forest model
#     oob_score: out-of-bag (OOB) samples
    rf_model = RandomForestClassifier(n_estimators=1000, max_features=2, oob_score=True, n_jobs =-1) 

#     select the features (name, parch, tickect and cabin have been removed)
    features = ["Sex","Pclass","SibSp","Embarked","Age","Fare"]
    
#     select the target
    target = ["Survived"]
    
#     train the model
#     ravel(): return a contiguous flattened array
    rf_model.fit(X=df_titanic_train[features], y=df_titanic_train[target].values.ravel())
    
    print("OOB ACCURACY:")
    print(rf_model.oob_score_)
    print()

#     sort features importances
    features_importances = zip(features, rf_model.feature_importances_)
    features_importances = list(features_importances)
    features_importances.sort(key=lambda x: x[1], reverse=True)   
 
    print("FEATURE IMPORTANCES:")
    for feature, importance in features_importances:
        print(feature, importance)
    print()
        
#     2. TEST THE MODEL
#     get training data 
    df_titanic_test = pd.read_csv("test.csv")
    
#     data preprocessing - fill nan
    df_titanic_test = df_titanic_test.fillna(df_titanic_test.median()[:"Age"], inplace=True)        
    df_titanic_test["Embarked"] = df_titanic_test["Embarked"].fillna("S")      
    df_titanic_test["Fare"] = df_titanic_test["Fare"].fillna(0)         
#     data preprocessing - convert string features to numeric values
    label_encoder = preprocessing.LabelEncoder()    
#     male = 1, female = 2
    df_titanic_test["Sex"] = label_encoder.fit_transform(df_titanic_test["Sex"])    
#     C = 0, Q = 1, S = 2
    df_titanic_test["Embarked"] = label_encoder.fit_transform(df_titanic_test["Embarked"])
    
#     get test predictions
    test_predictions = rf_model.predict(X=df_titanic_test[features])

#     set the submission data frame
    submission_data = pd.DataFrame({"PassengerId":df_titanic_test["PassengerId"], "Survived":test_predictions})
    
#     create the tutorial randomforest submission csv file
    submission_data.to_csv("Tutorial_RandomForest_Submission.csv", index=False) 
    
    print("Random Forest Submission csv file created.")
    
if __name__ == '__main__':
    main()