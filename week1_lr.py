# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
#prepare the environment
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

import sklearn
pd.show_versions()
#matplotlib.__version__
sklearn.__version__


# %%
get_ipython().system('ls')


# %%
#load the data
from sklearn.datasets import load_boston
boston_dataset = load_boston()
print(type(boston_dataset))


# %%
#explore the data
print(boston_dataset.keys())
print(boston_dataset['feature_names'])
print(boston_dataset['DESCR'])

# %% [markdown]
# Let's look at how the number of rooms (RM) affects the price (MEDV)
# In our linear regression, MEDV will be our Y value.  RM will be our X value.  

# %%
#prepare the data
from sklearn.model_selection import train_test_split
num_Rooms_Train, num_Rooms_Test, med_price_Train, med_Price_Test = train_test_split(boston_dataset.data[:,5].reshape(-1,1), boston_dataset.target.reshape(-1,1))
print(num_Rooms_Train.shape)
print(med_price_Train.shape)
print(num_Rooms_Test.shape)
print(num_Rooms_Test[0:10])


# %%
print(boston_dataset.data[:,5].shape)
print(boston_dataset.data[:,5])


# %%
#implement linear regression model
from sklearn.linear_model import LinearRegression
price_room = LinearRegression()
print(type(price_room))
price_room.fit (num_Rooms_Train,med_price_Train)


# %%
#predict on linear regression model

print(price_room.predict(num_Rooms_Test[0].reshape(-1,1)))
print(price_room.predict(np.array([7.564,3.543,2.450]).reshape(-1,1)))
print(num_Rooms_Test[0])
print([7.564])
print(price_room.predict(num_Rooms_Test[5].reshape(-1,1)))
print(price_room.predict(num_Rooms_Test[1:10]))
med_price_pred = price_room.predict(num_Rooms_Test)      


# %%
#Let's graph it:
#num_Rooms_Train, num_Rooms_Test, med_price_Train, med_Price_Test

import matplotlib.pyplot as plt
plt.scatter(num_Rooms_Train, med_price_Train, color = 'green')
plt.scatter(num_Rooms_Test, med_Price_Test, color = 'red')   
plt.scatter(num_Rooms_Test, med_price_pred, color = 'blue')  # The predicted temperatures of the same X_test input.
plt.plot(num_Rooms_Test, price_room.predict(num_Rooms_Test), color = 'gray')
plt.title('house price based on number of rooms')
plt.xlabel('number of rooms')
plt.ylabel('house price')
plt.show()


# %%
print('Coefficients: \n', price_room.coef_) 


# %%
from sklearn.metrics import mean_squared_error,r2_score

print('Mean squared error: %.2f' %mean_squared_error(med_Price_Test, med_price_pred))
rmse = np.sqrt(mean_squared_error(med_Price_Test, med_price_pred))
print('Root MSE: %.2f' %rmse)




# %%
# starter for group exercise


# %%
from sklearn.datasets import fetch_california_housing


# %%
cali_h = fetch_california_housing()


# %%
print(type(cali_h))


# %%
cali_h.keys()


# %%
print(cali_h.DESCR)


# %%
cali_features, cali_label = fetch_california_housing(return_X_y=True)


# %%
print(cali_features[0:5])


# %%
#pick a feature
#why are you picking this feature?


# %%
#split the features using numpy
#hint train = dataset[:-500]
#     test = dataset[-500:] 



# %%
#create and train a linear regressor


# %%
#print the coefficients, MSE, and RMSE


# %%
#optional:
#graph it


# %%
from sklearn.metrics import mean_squared_error,r2_scorefrom sklearn.metrics import mean_squared_error,r2_score

print('Mean squared error: %.2f' %mean_squared_error(diabetes_y_train, diabetes_y_pred))
rmse = np.sqrt(mean_squared_error(diabetes_y_train, diabetes_y_pred))
print('Root MSE: %.2f' %rmse)print('Mean squared error: %.2f' %mean_squared_error(diabetes_y_train, diabetes_y_pred))
rmse = np.sqrt(mean_squared_error(diabetes_y_train, diabetes_y_pred))
print('Root MSE: %.2f' %rmse)


