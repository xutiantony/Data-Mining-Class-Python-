# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:21:40 2023

@author: xutian
"""


import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/xutian/Downloads/Sheet4.CSV")

df_cleaned = df.dropna()

df_cleaned = df_cleaned.drop(columns = 'Name')

df_cleaned = pd.get_dummies(df_cleaned, columns=['Manuf', 'Type'])

Y = df_cleaned['Rating_Binary']
X= df_cleaned.drop(columns = 'Rating_Binary')


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

rf  = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [50,100,150,200],  
    'max_features': [3,4,5,6],
    'min_samples_leaf': [1, 2, 4] 
}

grid_search = GridSearchCV(rf, param_grid, cv=5, verbose=True, n_jobs=-1)

grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-validation Score: ", grid_search.best_score_)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

from sklearn.ensemble import GradientBoostingClassifier
GB  = GradientBoostingClassifier()
grid_search = GridSearchCV(GB, param_grid, cv=5, verbose=True, n_jobs=-1)

grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-validation Score: ", grid_search.best_score_)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)