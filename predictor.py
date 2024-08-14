import pandas as pd
from utils import load_object,clusters

scaler = load_object('artifacts','scaler.pkl')
cluster_finder = load_object('artifacts','kmeans.pkl')
model_cluster0 =load_object('artifacts/models','Gradient Boosting_cluster0.pkl')
model_cluster1 = load_object('artifacts/models','Random Forest_cluster1.pkl')
model_cluster2 = load_object('artifacts/models','Ada Boost_cluster2.pkl')

def predict(data):
    cluster = cluster_finder.predict(data)
    scaled_data = scaler.fit_transform(data)
    if cluster==0:
        prediction = model_cluster0.predict(scaled_data)
    elif cluster==1:
        prediction = model_cluster1.predict(scaled_data)
    elif cluster==2:
        prediction = model_cluster2.predict(scaled_data)
    return prediction
