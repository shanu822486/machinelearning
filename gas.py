
# coding: utf-8

# # **Predicting Gas Consumption Value using Linear Regression**

# ## Import Libraries
# #### _Import the usual libraries_

# In[131]:


import pandas as pd , numpy as np, pickle
import matplotlib.pyplot as plt,seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn import metrics  
#from pylab import rcParams
import seaborn as sb
#rcParams['figure.figsize'] = 7, 7
sb.set_style('darkgrid')


# ## Dataset Information
# #### _Gas_
# 
# 

# ## Get the Data
# 
# ** Use pandas to read petrol_consumption.csv as a dataframe called gas.**

# In[128]:


gas = pd.read_csv("petrol_consumption.csv")  
print(gas.shape)
gas.head()
#print(gas.describe())


# ## Data Cleansing
# #### if there, _removing the special chars & converting the corresponding object attributes to float_ 

# In[120]:


gas.columns = [ 'gas_tax','avg_income','Paved_highways','pop_dl','consumption']
#gas.info()


# ## Exploratory Data Analysis
# 

# In[121]:


g = sns.lmplot(x="gas_tax", y="consumption", data=gas)
g = sns.lmplot(x="avg_income", y="consumption", data=gas)
g = sns.lmplot(x="Paved_highways", y="consumption", data=gas)
g = sns.lmplot(x="pop_dl", y="consumption", data=gas)


# # Train Test Split
# 
# ** Split your data into a training set and a testing set.**

# In[122]:


droplst = ['Paved_highways','avg_income','consumption']
X = gas.drop(droplst,axis=1)
y = gas['consumption']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# # Train a Model
# 
# Now its time to train a Linear Regression. 
# 
# **Call the LinearRegression() model from sklearn and fit the model to the training data.**

# In[132]:


model = LinearRegression()  
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)  
scor  = metrics.r2_score(y_test, y_pred)
print("SCORE=",scor)
predictiondf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(predictiondf)  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
g = sns.lmplot(x="Actual", y="Predicted", data=predictiondf)
# plt.scatter(y_test,y_pred,c='m')
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')

filename = 'gaslr.pkl'
pickle.dump(model, open(filename, 'wb'))
print("PICKLE FILE GENERATED SUCCESSFULLY\n\n\n\n")
print("PROGRAM ENDS SUCCESSFULLY")

