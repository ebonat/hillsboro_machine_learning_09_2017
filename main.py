
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

def titanic_preprocessing(df_titanic):
    df_titanic = df_titanic.fillna(df_titanic.median()[:"Age"], inplace=True)    
    df_titanic["Embarked"] = df_titanic["Embarked"].fillna("S")      
    label_encoder = preprocessing.LabelEncoder()
    df_titanic["Sex"] = label_encoder.fit_transform(df_titanic["Sex"])
    df_titanic["Embarked"] = label_encoder.fit_transform(df_titanic["Embarked"])
    df_titanic["Fare"] = df_titanic["Fare"].fillna(0)     
    return df_titanic

def main():
    
#     TRAIN THE MODEL ---------------------------------------------------------------------
    titanic_train = pd.read_csv("train.csv")
    print(titanic_train)
    print()
    
#     PREPROCESSING TRAIN DATA -----------------------------------------------------------------
#     new_age_var = np.where(titanic_train["Age"].isnull(), 24, titanic_train["Age"])     
#     titanic_train["Age"] = new_age_var     
#     titanic_train = titanic_train.fillna(titanic_train.median()[:"Age"], inplace=True)    
#     titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")      
#     label_encoder = preprocessing.LabelEncoder()
#     titanic_train["Sex"] = label_encoder.fit_transform(titanic_train["Sex"])
#     titanic_train["Embarked"] = label_encoder.fit_transform(titanic_train["Embarked"])
    
#     PREPROCESSING TRAIN DATA -----------------------------------------------------------------
    titanic_train = titanic_preprocessing(titanic_train)
    
    rf_model = RandomForestClassifier(n_estimators=1000, max_features=2, oob_score=True) 
    
    features = ["Sex","Pclass","SibSp","Embarked","Age","Fare"]
    
    target = ["Survived"]
    
#     train the model
    rf_model.fit(X=titanic_train[features], y=titanic_train[target].values.ravel())

    titanic_train.to_csv("train_clean.csv")

    print("OOB ACCURACY: ")
    print(rf_model.oob_score_)
    print()
    
    print("FEATURE IMPORTANCES:")
    for feature, imp in zip(features, rf_model.feature_importances_):
        print(feature, imp)

#         TEST THE MODEL -------------------------------------------------------------------------------

    titanic_test = pd.read_csv("test.csv")
    
#     PREPROCESSING TEST DATA -----------------------------------------------------------------
#     new_age_var = np.where(titanic_test["Age"].isnull(), 24, titanic_test["Age"])     
#     titanic_test["Age"] = new_age_var     
#     titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")      
#     label_encoder = preprocessing.LabelEncoder()
#     titanic_test["Sex"] = label_encoder.fit_transform(titanic_test["Sex"])
#     titanic_test["Embarked"] = label_encoder.fit_transform(titanic_test["Embarked"])    
#     titanic_test["Fare"] = titanic_test["Fare"].fillna(0)     
    
#     PREPROCESSING TEST DATA -----------------------------------------------------------------
    titanic_test = titanic_preprocessing(titanic_test)
    
    titanic_train.to_csv("test_clean.csv")
    
    test_predictions = rf_model.predict(X=titanic_test[features])

    submission_data = pd.DataFrame({"PassengerId":titanic_test["PassengerId"], "Survived":test_predictions})
    
    submission_data.to_csv("Tutorial_RandomForest_Submission.csv", index=False) 

if __name__ == '__main__':
    main()