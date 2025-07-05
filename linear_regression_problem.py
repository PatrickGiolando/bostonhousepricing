import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# %matplotlib inline

from sklearn.datasets import fetch_california_housing

# lol boston is racist
housing = fetch_california_housing(as_frame=True)
boston = housing.frame

dataset = boston.copy()
# dataset['Price'] = dataset['MedHouseVal']# * 100000  # Convert to dollars




# Set up Training and Test Set
# ---------------------------------------
#dataset.head()

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Model Training
# ----------------------------------------
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
# print(regression.coef_)

# Model Evaluation
# ----------------------------------------
reg_pred = regression.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score



# pickle
pickle.dump(regression, open('regmodel.pkl', 'wb'))
# pickle_model = pickle.load(open('regmodel.pkl', 'rb'))
# pickle_pred = pickle_model.predict(X_test)





# plt.plot(y_test, reg_pred, 'o')
# plt.show()