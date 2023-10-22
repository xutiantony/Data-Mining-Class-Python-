# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 06:15:39 2023

@author: Tony
"""
import pandas as pd
import numpy as np

df = [['black', 1,1],['blue',0,0],['blue',-1,-1]]

df = pd.DataFrame(df, columns=['Color', 'x1', 'x2'])



from sklearn.neighbors import KNeighborsClassifier


X = df.iloc[:,1:3]
y = df['Color']

knn = KNeighborsClassifier(n_neighbors=2)
model = knn.fit(X,y)

new_obs = [[0.1,0.1]]
model.predict(new_obs)
model.predict_proba(new_obs)