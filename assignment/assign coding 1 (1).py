# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#0

import pandas as pd
df = pd.read_csv("C:/Users/xutian/Downloads/ToyotaCorolla.CSV")

X = df.iloc[:,3:12]
Y = df.iloc[:,2] 

#1
from statsmodels.tools.tools import add_constant
X1 = add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns
  
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(X1.columns)):
    vif_data.loc[vif_data.index[i],"VIF"] = variance_inflation_factor(X1.values, i)

print(vif_data)

X = X.drop(columns=['Cylinders'])

#2
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns=X.columns) 

#3
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size=0.35, random_state=662)

#4

from sklearn.linear_model import LinearRegression
lm2 = LinearRegression()
model2 = lm2.fit(X_train,Y_train)
y_test_pred = model2.predict(X_test)

from sklearn.metrics import mean_squared_error
lm2_mse = mean_squared_error(Y_test, y_test_pred)
print("Test MSE using validation set approach = "+str(lm2_mse))


#5


from sklearn.linear_model import Ridge

# Run ridge regression with penalty equals to 1
ridge = Ridge(alpha=1)
ridge_model = ridge.fit(X_train,Y_train)

y_test_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(Y_test, y_test_pred)
print("Test MSE using ridge with penalty of 1 = "+str(ridge_mse))



#6
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1)
lasso_model = lasso.fit(X_train,Y_train)

y_test_pred4 = lasso_model.predict(X_test)

lasso_mse = mean_squared_error(Y_test, y_test_pred4)
print("Test MSE using lasso with penalty of 1 = "+str(lasso_mse))

#7

for i in [10,100,1000,10000]:
    lasso = Lasso(alpha=i)
    lasso_model = lasso.fit(X_train,Y_train)
    y_test_pred5 = lasso_model.predict(X_test)
    print('Alpha = ',i,' / MSE =',mean_squared_error(Y_test, y_test_pred5))
    


for i in [10,100,1000,10000]:
    ridge = Ridge(alpha=i)
    ridge_model = ridge.fit(X_train,Y_train)

    y_test_pred = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(Y_test, y_test_pred)
    print("Test MSE using ridge with penalty of 1 = "+str(ridge_mse))

feature_names = X.columns.tolist()
coefficients = lasso_model.coef_


for feature, coef in zip(feature_names, coefficients):
    print(f'{feature}: {coef}')

feature_names = X.columns.tolist()
coefficients = ridge_model.coef_
for feature, coef in zip(feature_names, coefficients):
    print(f'{feature}: {coef}')



