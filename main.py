from data_ingestion import data_ingestor
from utils import train_test_splitter
from components.model_finder import ModelTrainer
data = data_ingestor.data_fetcher('default of credit card clients.xls')
X=data.drop(['default payment next month'],axis=1)
y=data['default payment next month']

X_train,y_train,X_test,y_test=train_test_splitter(X,y)

modeltrainer = ModelTrainer()
print(modeltrainer.initiate_model_training(X_train,y_train,X_test,y_test))