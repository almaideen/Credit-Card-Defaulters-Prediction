import os
from data_ingestion import data_ingestor
from utils import train_test_splitter
from components.model_finder import ModelTrainer
from utils import scaler, clusters, save_object
from sklearn.cluster import KMeans

data = data_ingestor.data_fetcher('default of credit card clients.xls')
X=data.drop(['default payment next month'],axis=1)
y=data['default payment next month']

#clustering
number_of_clusters=clusters(X)
kmeans = KMeans(n_clusters=number_of_clusters,init='k-means++',random_state=5)
y_kmeans=kmeans.fit_predict(X)
X['cluster']=y_kmeans
X['label']=y

#training model based on cluster data
for i in range(0,number_of_clusters):
    cluster_data=X[X['cluster']==i]

    #Prepare dependant and Indepandant feature
    cluster_features=cluster_data.drop(['cluster','label'],axis=1)
    cluster_label=cluster_data['label']

    #Splitting into train test split:
    X_train,X_test,y_train,y_test=train_test_splitter(cluster_features,cluster_label)
    X_train_scaled,X_test_scaled=scaler(X_train,X_test)
    modeltrainer = ModelTrainer()
    model_report,best_model_name,best_model=modeltrainer.initiate_model_training(X_train_scaled,y_train,X_test_scaled,y_test)
    
    print(f"Score Report for Cluster {i}:")
    print(model_report)

    path=os.path.join("artifacts/models",best_model_name+"_cluster"+str(i)+".pkl")
    save_object(path,best_model)
    



'''
#Scaling
X_Scaled = scaler(X)
data_updated=X_Scaled
data_updated['cluster']=y_kmeans
data_updated['label']=y

#Updated X and y
X=data_updated.drop(['label'],axis=1)
y=data_updated['label']

print(X.head())


X_train,y_train,X_test,y_test=train_test_splitter(X,y)
modeltrainer = ModelTrainer()
print(modeltrainer.initiate_model_training(X_train_scaled,y_train,X_test_scaled,y_test))'''