import sys
import os
import pickle
from sklearn.model_selection import train_test_split
from logger import logging
from exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_test_splitter(x,y):
    try:
        X_train,y_train,X_test,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
        logging.info("Created Train and Test data")
        return X_train,y_train,X_test,y_test
    except Exception as e:
        logging.info("e")
        raise CustomException(e,sys)
    
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            logging.info(f"{obj} pickle file created!")
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        logging.info(f"Error occured while creating {obj} pickle file!")
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(models)):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            grid_search=GridSearchCV(model,para,cv=5)
            grid_search.fit(X_train,y_train)
            model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score=accuracy_score(y_train,y_train_pred)
            test_model_score=accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = train_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys) 