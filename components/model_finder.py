import os
import sys
import stats
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XgboostClassifier

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,X_train,y_train,X_test,y_test):
        try:
            models={
                'Logistic Regression':LogisticRegression(),
                'SVC':SVC(),
                'Decision Tree':DecisionTreeClassifier(),
                'Random Forest':RandomForestClassifier(),
                'Ada Boost':AdaBoostClassifier(),
                'Gradient Boosting':GradientBoostingClassifier(),
                'XGBoost': XgboostClassifier()

            }
            params={
                'Logistic Regression':{
                    'penalty':['l1', 'l2', 'elasticnet', None],
                    'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                },
                'SVC':{
                    'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'degree':[1,2,3,4,5],
                    'gamma':['scale', 'auto']
                },
                'Decision Tree':{
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'splitter':['best', 'random'],
                    'max_depth':[1,2,3,4,5],
                    'min_samples_split':[1,2,3,4,5]
                },
                'Random Forest':{
                    'n_estimators':[10,20,40,60,80,100],
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'max_depth':[1,2,3,4,5],
                    'min_samples_split':[1,2,3,4,5],
                    'max_features':['sqrt', 'log2', None]
                },
                'Ada Boost':{
                    'n_estimators':[10,20,40,60,80,100],
                    'learning_rate':[0.001,0.01,0.1,0.5],
                    'algorithm':['SAMME', 'SAMME.R']
                },
                'Gradient Boosting':{
                    'loss':['log_loss', 'exponential'],
                    'learning_rate':[0.001,0.01,0.1,0.5],
                    'n_estimators':[10,20,40,60,80,100],
                    'subsample':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    'criterion':['friedman_mse', 'squared_error']
                },
                'XGBoost':{
                    'max_depth': stats.randint(3, 10),
                    'learning_rate': stats.uniform(0.01, 0.1),
                    'subsample': stats.uniform(0.5, 0.5),
                    'n_estimators':stats.randint(50, 200)
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            best_model_score= max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found!")
            logging.info("Best model found!")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            rsquare = r2_score(y_test,predicted)
            print(best_model,rsquare)
        
        except Exception as e:
            raise CustomException(e,sys)
