import sys
import os
import pickle
import numpy as np
import kneed
from sklearn.model_selection import train_test_split
from logger import logging
from exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

def train_test_splitter(x,y):
    try:
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
        logging.info("Created Train and Test data")
        return X_train,X_test,y_train,y_test
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

def load_object(file_path,file_name):
    try:
        path = os.path.join(file_path,file_name)
        file=open(path,'rb')
        loaded_object = pickle.load(file)
        logging.info(f"{file_name} loaded successfully!")
        return loaded_object
    except Exception as e:
        raise CustomException(e,sys)


def scaler(X_train,X_test):
    stdscaler = StandardScaler()
    X_train_scaled = stdscaler.fit_transform(X_train)
    X_test_scaled=stdscaler.transform(X_test)
    path=os.path.join("artifacts","scaler.pkl")
    save_object(file_path=path,obj=stdscaler)
    logging.info("Scaler file is saved!")
    return (X_train_scaled,X_test_scaled)

def clusters(data):
    wcss=[]
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i,init='k-means++',random_state=5)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    path=os.path.join("artifacts","kmeans.pkl")
    save_object(file_path=path,obj=kmeans)
    logging.info("KMeans file is saved!")
    KL = KneeLocator(range(1,11),wcss,curve='convex',direction='decreasing')
    logging.info(f"Number of clusters found: {KL.knee}")
    print(f"Number of clusters found: {KL.knee}")
    return KL.knee

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for i in range(len(models)):
            model = list(models.values())[i]
            model_names=list(models.keys())
            #para=param[list(models.keys())[i]]
            #grid_search=GridSearchCV(model,para,cv=5)
            #grid_search.fit(X_train,y_train)
            #model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score=accuracy_score(y_train,y_train_pred)
            test_model_score=accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = train_model_score,test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys) 
    
