
from random_forest_class import RandomForestClass
import config

def main():
#     create class instant
    random_forest_class = RandomForestClass()

#     get the data into the data frame
    df_titanic_train = random_forest_class.read_file_csv(config.TRAIN_FILE)

#     create the random forest model
    rf_model = random_forest_class.get_rf_model(config.RF_N_ESTIMATORS, config.RF_MAX_FEATURES) 
    
#     fit the random forest model with training data 
    rf_model = random_forest_class.rf_model_fit(rf_model, df_titanic_train[config.FEATURES], df_titanic_train[config.TARGET].values.ravel())

#     calculate out-of-bag score
    oob_score_value = random_forest_class.calculate_oob_accuracy(rf_model)
    print("OOB ACCURACY:")
    print(oob_score_value)
    print()

#     get list features importances score
    features_importances = random_forest_class.get_features_importances(config.FEATURES, rf_model)
    print("FEATURE IMPORTANCES:")
    for feature, importance in features_importances:
        print(feature, importance)
    print()

#     df_titanic_test = random_forest_class.read_file_csv(config.TEST_FILE)
        
if __name__ == '__main__':
    main()