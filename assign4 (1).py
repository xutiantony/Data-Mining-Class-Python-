# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:47:06 2023

@author: Tony
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/xutian/Downloads/OnlineNews.csv")

df=df.drop(['url','timedelta'],axis=1)

X = df.drop(['popularity'],axis=1)
Y = df['popularity']

X = pd.get_dummies(X, columns = ['data_channel','weekdays','is_weekend' ])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, Y, test_size = 0.3, random_state = 5)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for i in  (50,100,150,200):
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean', weights='uniform')
    model = knn.fit(X_train,y_train)
    y_test_pred = model.predict(X_test)
    print("Accuracy score using k-NN with ",i," neighbors = "+str(accuracy_score(y_test, y_test_pred)))
    


from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.05) 
model = ls.fit(X_std,Y)

dfa=pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])

dfa[dfa['coefficient'] != 0]

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_std)


from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
#model 1
X_train, X_test, y_train, y_test = train_test_split(X_std, Y, 
                                                    test_size=0.3, 
                                                    random_state=5)

knn = KNeighborsClassifier(n_neighbors=200)
model = knn.fit(X_train,y_train)
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

training_time = end_time - start_time

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Model Training Time: {training_time:.2f} seconds")


#model 2
X_lasso = X[['kw_avg_avg', 'LDA_02','data_channel_Entertainment', 'data_channel_Social Media', 'data_channel_World', 'is_weekend_0']]
scaler = StandardScaler()
scaler.fit(X_lasso)
X_lasso = scaler.transform(X_lasso)



X_train, X_test, y_train, y_test = train_test_split(X_lasso, Y, 
                                                    test_size=0.3, 
                                                    random_state=5)

knn = KNeighborsClassifier(n_neighbors=200)
model = knn.fit(X_train,y_train)
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

training_time = end_time - start_time

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Model Training Time: {training_time:.2f} seconds")


#model3
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, 
                                                    test_size=0.3, 
                                                    random_state=5)

knn = KNeighborsClassifier(n_neighbors=200)
model = knn.fit(X_train,y_train)
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

training_time = end_time - start_time

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Model Training Time: {training_time:.2f} seconds")






