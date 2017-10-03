
import pandas as pd
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

class RandomForestClass(object):

    def __init__(self):
        pass

    def read_file_csv(self, file_path_name, file_encoding=None):       
        try:
            if file_encoding is None:
                df_file = pd.read_csv(file_path_name)
            else:
                df_file = pd.read_csv(file_path_name, encoding=file_encoding)
        except Exception as ex:
            print( "An error occurred: {}".format(ex))      
        return df_file
    
    def titanic_data_preprocessing(self, df_titanic):     
        try:
            df_titanic = df_titanic.fillna(df_titanic.median()[:FeatureColumn.Age.value], inplace=True)                        
            df_titanic[FeatureColumn.Embarked.value] = df_titanic[FeatureColumn.Embarked.value].fillna("S")      
            df_titanic[FeatureColumn.Fare.value] = df_titanic[FeatureColumn.Fare.value].fillna(0)     
            label_encoder = preprocessing.LabelEncoder()
            df_titanic[FeatureColumn.Sex.value] = label_encoder.fit_transform(df_titanic[FeatureColumn.Sex.value])
            df_titanic[FeatureColumn.Embarked.value] = label_encoder.fit_transform(df_titanic[FeatureColumn.Embarked.value])            
        except Exception as ex:
            print( "An error occurred: {}".format(ex))      
        return df_titanic

    def get_rf_model(self, number_estimators, maximun_features):      
        try:
            rf_model = RandomForestClassifier(n_estimators=number_estimators, max_features=maximun_features, oob_score=True) 
        except Exception as ex:
            print( "An error occurred: {}".format(ex))      
        return rf_model
    
    def rf_model_fit(self, rf_model, X_train, y_train):          
        try:
            rf_model.fit(X=X_train, y=y_train)
        except Exception as ex:
            print( "An error occurred: {}".format(ex))      
        return rf_model
    
    def calculate_oob_accuracy(self, rf_model):
        try:
            oob_score_value = rf_model.oob_score_            
        except Exception as ex:
            print( "An error occurred: {}".format(ex))      
        return oob_score_value
    
    def get_features_importances(self, features, rf_model):
        try:
            features_importances = zip(features, rf_model.feature_importances_)
            features_importances = list(features_importances)
            features_importances.sort(key=lambda x: x[1], reverse=True)         
        except Exception as ex:
            print( "An error occurred: {}".format(ex))      
        return features_importances
        