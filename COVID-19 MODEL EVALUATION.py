#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install -U scikit-learn scipy matplotlib')
get_ipython().system('{sys.executable} -m pip install xgboost')
get_ipython().system('pip install optuna')
get_ipython().system('pip install numpy')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install -U statsmodels')

#import necessary packages

import sys 
import optuna
import plotly 


import numpy as np 
import pandas as pd 
import xgboost as xgb 
import seaborn as sns  
import plotly.express as px 
import matplotlib.pyplot as plt 
from xgboost import XGBClassifier


from pathlib import Path 
from pyspark.sql import SparkSession
from pyspark.sql import functions as func 
from pyspark.sql.types import IntegerType, DoubleType, StringType, FloatType
from pyspark.ml.classification import DecisionTreeClassifier 
from pyspark.ml.feature import StringIndexer, VectorAssembler 
from xgboost import XGBClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split ,KFold,StratifiedKFold 
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.metrics import classification_report ,confusion_matrix,ConfusionMatrixDisplay 
from sklearn.metrics import roc_curve, auc ,roc_auc_score #There are several limitations when using this method with binaries, multiclass, and multi - label categorization.
from sklearn.cluster import KMeans # K-means++ uses sampling to choose the initial cluster centroids based on an actual probability density function of the contribution of each point to the total inertia.
from numpy import nan #Not a number is referred to as NaN. It is employed to symbolise undefined values. In a dataset, it is also employed to represent any missing values.



# In[2]:


#create spark context
sqlCtx = SparkSession.builder.getOrCreate()


# In[3]:


#Here i wanted to see general data to have an idea about data type

df = sqlCtx.read.option('header','true').csv('covid.csv', inferSchema =True)

df.show()
df.dtypes


# In[4]:


#I also wanted to read the data as a dataframe to have better inshight
 
df = pd.read_csv("covid.csv")
df.head(20)


# In[5]:


#Here i dropped the three columns from data to see only countries and cases day one day. We can also do this with
#the help of excel tool. 

dropcol = df


# In[6]:


dropcol.drop(['Lat','Long', 'Province/State'], axis=1, inplace=True)
df.head()


# In[7]:


#we can have better ide about the dataset. In this case we can understand the number of columns and rows in the dataset.
#Dataset size and information about data with the ghelp of dataframe. 

print("dataset shape  : ",df.shape)
print("dataset rows   : ",df.shape[0])
print("dataset columns: ",df.shape[1])
print("dataset size   : ",df.size)
print(df.info())


# In[8]:


#here i checked the missing values and wanted to see total of missing values.

print(df.isnull().sum())
print()
print("Missing values = ",df.isnull().sum().sum())


# In[9]:


#i wanted to count the data to be able to identify top 3 countries. It was hard to tell which countries are top 3.

dataFrame = dropcol.sum(axis = 1)
print("\nCounting rows...\n",dataFrame)


# In[10]:



df.values.tolist()


# In[11]:


# i also wanted to see how many total case around the world so far. 
my_list = dataFrame
print ("The sum of my_list is", sum(my_list))


# # I organised the data in excel
# 
# ### Dataset organized in the excel tool. According to total case numbers, top 3 country was Brazil, India and US. 
# ###

# In[12]:


df = pd.read_csv("top3.csv")
df.head()


# In[13]:


df.var()


# In[14]:


df.var(axis=1)


# # CREATING LINEER REGRATION MODEL

# In[15]:


get_ipython().system('pip install seaborn # Visualizing library')

import pandas as pd
# pandas is a software library written for the Python programming language for data manipulation and analysis.
# In this code we use it to show scatter plots. 
import seaborn as sb 
from matplotlib import pyplot as plt # We use matplotlib for create axe and figures to plot data
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col # For use the column name of the dataframe in pyspark
from pyspark.sql import SparkSession


# In[16]:


top3 = sqlCtx.read.option('header','true').csv('top3.csv', inferSchema=True)
top3.show()
print(top3.dtypes)


# In[17]:


top3 = sqlCtx.read.option('header','true').options(delimiter=",").csv('top3.csv')
top3.show()
print(top3.dtypes)

for _ in top3.columns:
    top3 = top3.withColumn(_,col(_).cast(DoubleType()))
print(top3.dtypes)


# In[18]:


top3_pandas_dataframe = top3.toPandas()

fig, ax = plt.subplots(figsize=(10,10))
sb.heatmap(top3_pandas_dataframe.corr(), cmap="Blues", annot=True, ax=ax)


# # LINEAR REGRSESSION

# In[19]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[20]:


df.head() 


# In[21]:


df.shape


# In[22]:


df = pd.read_csv('top3.csv')
df_1 = df.transpose()
print(df_1)


# In[23]:


df = pd.read_csv('top3.csv')


# In[24]:


df.head(20)


# In[25]:


df.shape


# In[26]:


df.plot.scatter(x='Weeks', y='Brazil', title='Scatterplot of weeks and Brazil');
df.plot.scatter(x='Weeks', y='India', title='Scatterplot of weeks and India');
df.plot.scatter(x='Weeks', y='US', title='Scatterplot of weeks and US');


# In[27]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:




sns.pairplot(df, y_vars=['Brazil', 'India', 'US'], x_vars= 'Weeks', height=7, aspect = 0.7, kind='reg')                  


# In[29]:


print(df.corr())


# In[30]:


print(df.describe())


# # Brazil variance score 

# In[31]:


#User:AmiyaRanjanRout
#08 Oct, 2021
#Title:Calculate the average, variance and standard deviation in Python using NumPy
#Link:https://www.geeksforgeeks.org/calculate-the-average-variance-and-standard-deviation-in-python-using-numpy/


#defining y

feature_BRAZIL_cols = ['Brazil']
y = df[feature_BRAZIL_cols]
y = df[['Brazil']]
y


# In[32]:


#Defining X

X = df['Weeks']
X


# In[33]:


from sklearn.model_selection import train_test_split

#training and testing X,y 

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

print(y_train.shape)
print(X_train.shape)
print(y_test.shape)
print(X_test.shape)


# In[34]:


from sklearn.linear_model import LinearRegression

#linear regression
#fittig
linreg = LinearRegression()
linreg.fit(y_train, X_train)


# In[35]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
  
# creating training and test sets from X and Y

from sklearn.model_selection import train_test_split
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.4,
                                                    random_state=1)
  
# here i created a linear regression object
reg = linear_model.LinearRegression()
  
# utilising the training sets, train the model
reg.fit(y_train, X_train)
  
# regression coefficients
print('Coefficients: ', reg.coef_)
  
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(y_test, X_test)))
  
# plot for residual error
  
# here i setted the plot style
plt.style.use('fivethirtyeight')
  
# plotting the residual errors in the training and test data
plt.scatter(reg.predict(y_train), reg.predict(y_train) - X_train,
            color = "green", s = 10, label = 'Train data')
  
plt.scatter(reg.predict(y_test), reg.predict(y_test) - X_test,
            color = "blue", s = 10, label = 'Test data')
  
# plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
# plotting legend
plt.legend(loc = 'upper right')
  
# for the plot title
plt.title("Brazil Residual errors")
  
# showing the plot
plt.show()


# # India variance score

# In[36]:


feature_INDIA_cols = ['India']
I = df[feature_INDIA_cols]
I = df[['India']]
I


# In[37]:


X = df['Weeks']
X


# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, I_train, I_test = train_test_split(X,I,random_state=1)

print(I_train.shape)
print(X_train.shape)
print(I_test.shape)
print(X_test.shape)


# In[39]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(I_train, X_train)


# In[40]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics


from sklearn.model_selection import train_test_split
I_train, I_test, X_train, X_test = train_test_split(I, X, test_size=0.4,
                                                    random_state=1)
  
reg = linear_model.LinearRegression()
  
reg.fit(I_train, X_train)
  
print('Coefficients: ', reg.coef_)
  
print('Variance score: {}'.format(reg.score(I_test, X_test)))
  
# plot for residual error
  
plt.style.use('fivethirtyeight')
  
plt.scatter(reg.predict(I_train), reg.predict(I_train) - X_train,
            color = "green", s = 10, label = 'Train data')
  
plt.scatter(reg.predict(I_test), reg.predict(I_test) - X_test,
            color = "blue", s = 10, label = 'Test data')
  
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
plt.legend(loc = 'upper right')
  
plt.title("India Residual errors")
  
plt.show()


# # US variance score

# In[41]:


feature_US_cols = ['US']
U = df[feature_US_cols]
U = df[['US']]
X = df['Weeks']


# In[42]:


from sklearn.model_selection import train_test_split

X_train, X_test, U_train, U_test = train_test_split(X,U,random_state=1)

print(U_train.shape)
print(X_train.shape)
print(U_test.shape)
print(X_test.shape)


# In[43]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(U_train, X_train)


# In[44]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
  

from sklearn.model_selection import train_test_split
U_train, U_test, X_train, X_test = train_test_split(U, X, test_size=0.4,
                                                    random_state=1)
  
reg = linear_model.LinearRegression()
  
reg.fit(U_train, X_train)
  
print('Coefficients: ', reg.coef_)
  
print('Variance score: {}'.format(reg.score(U_test, X_test)))
    
plt.style.use('fivethirtyeight')
  
plt.scatter(reg.predict(U_train), reg.predict(U_train) - X_train,
            color = "green", s = 10, label = 'Train data')
  
plt.scatter(reg.predict(U_test), reg.predict(U_test) - X_test,
            color = "blue", s = 10, label = 'Test data')
  
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
plt.legend(loc = 'upper right')
  
plt.title("US Residual errors")
  
plt.show()


# # Variance Scores
# 
# #### US: 0.9434838241430802
# #### Brazil: 0.980267362747466
# #### India: 0.947855835900343
# 
# ###### Brazil has the highest variance. Brazil and its linear regression model will be selected for further steps.

# # b) Clustering

# In[45]:


#link: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
#Title: The Most Comprehensive Guide to K-Means Clustering You’ll Ever Need
#publish date: August 19, 2019
#Author: Pulkit Sharma

import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt


# In[46]:


data = pd.read_csv('top3.csv')
data.head()


# In[47]:


dropdata = data

dropdata.drop('US',axis=1,inplace=True)
dropdata.drop('India',axis=1,inplace=True)
data.head()


# ## Inertia (Elbow Method)

# In[48]:


#K-Means++ Elbow Method


# In[49]:


get_ipython().system('pip install sklearn')

# importing necessery libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


# In[50]:


#reading data
data=pd.read_csv("top3.csv")
data.head()


# In[51]:


#dropping data 
dropdata = data

dropdata.drop('US',axis=1,inplace=True)
dropdata.drop('India',axis=1,inplace=True)
data.head()


# In[52]:


#pullling out statistics of the new dataset.
data.describe()


# In[53]:


#This magnitude discrepancy might be problematic as K-Means is a distance-based method.
#thats why i brought all the variables to same magnitue by standardizing.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# to be able to see statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[54]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=3, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[55]:


# here we are seeing inertia on the fitted data
kmeans.inertia_


# In[56]:


#fitting multiple k-means algorithms and storing the values in an empty list
#i took the range 20 for this analyse.

SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

#converting the results into a dataframe and plotting them
#in that case we can define the cluster number.
#According to the graph, the cluster number is 4.

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[57]:


#according to the elbow method I defined the numebr of cluster as 4.
model = KMeans(n_clusters=4)

#Here, I scalled the data to normalize it. This step is significant for good results.

model = model.fit(scale(data))


print(model.labels_)


# In[58]:


X = dropdata[["Weeks","Brazil"]]
#Visualising the data points here.
plt.figure(figsize=(8, 6))
plt.scatter(X["Weeks"],X["Brazil"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[59]:


#I choose 4 due to the elbow model. 
#here i alsa would like to know how many items fit in a cluster.

kmeans = KMeans(n_clusters = 4, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[60]:


#here i selected the number of cluster.
#and selected random centroids for each cluster.
K=4

# Select random observation as centroids
#everytime I run this programme i could get different centroits. 
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Brazil"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Brazil"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[61]:


#implementing some conditions for algorithm.
#first i assigned all the points to the closest cluster centroids.
#after that recomputed centroids of newly formed clusters.
#and repeted for those steps.
#values might come different eveytime. 
#When the centroids do not change after two iterations, I halt the training. 
#The difference between the centroids from the previous iteration and the current iteration is 
#what I use to calculate the diff, which we previously defined as 1.

diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Brazil"]-row_d["Brazil"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Brazil","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Brazil'] - Centroids['Brazil']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Brazil","Weeks"]]


# In[62]:


#i also wanted to see the clasters with Centroids. 

color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Brazil"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Brazil"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # c) Graph Analytics
# ### Neighbouring csv file created with excel based on my geographical knowladge. I will demonstrate the graphical view below.

# In[63]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[64]:


#linear regression of neighbour countries.

sns.pairplot(data, y_vars=['Uruguay', 'Argentina', 'Paraguay', 'Bolivia',
                           'Peru', 'Colombia','Venezuela','Guyana', 'Suriname' ], x_vars='Weeks' , 
             height=7, aspect = 0.7, kind='reg') 


# # Uruguay

# In[65]:


dropdata = data

dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[66]:


data.describe()


# In[67]:


# standardizing the data
from sklearn.preprocessing import StandardScaler
Uruguay_scaler = StandardScaler()
data_scaled = Uruguay_scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[68]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=4, init='k-means++')

# fitting the k means on the scaled data
kmeans.fit(data_scaled)


# In[69]:


kmeans.inertia_


# In[70]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[71]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[72]:


X = dropdata[["Weeks","Uruguay"]]
#Visualise data points
plt.scatter(X["Weeks"],X["Uruguay"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[73]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Uruguay"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Uruguay"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[74]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Uruguay"]-row_d["Uruguay"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Uruguay","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Uruguay'] - Centroids['Uruguay']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Uruguay","Weeks"]]


# In[75]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Uruguay"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Uruguay"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Argentina

# In[76]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[77]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[78]:


data.describe()


# In[79]:


# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[80]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=4, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[81]:


kmeans.inertia_


# In[82]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[83]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[84]:


X = dropdata[["Weeks","Argentina"]]
#Visualise data points
plt.scatter(X["Weeks"],X["Argentina"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[85]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Argentina"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Argentina"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[86]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Argentina"]-row_d["Argentina"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Argentina","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Argentina'] - Centroids['Argentina']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Argentina","Weeks"]]


# In[87]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Argentina"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Argentina"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Paraguay

# In[88]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[89]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[90]:


data.describe()


# In[91]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[92]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[93]:


kmeans.inertia_


# In[94]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[95]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[96]:


X = dropdata[["Weeks","Paraguay"]]
#Visualise data points
plt.figure(figsize=(8, 6))
plt.scatter(X["Weeks"],X["Paraguay"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[97]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Paraguay"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Paraguay"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[98]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Paraguay"]-row_d["Paraguay"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Paraguay","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Paraguay'] - Centroids['Paraguay']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Paraguay","Weeks"]]


# In[99]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Paraguay"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Paraguay"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Bolivia

# In[100]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[101]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[102]:


data.describe()


# In[103]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[104]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[105]:


kmeans.inertia_


# In[106]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[107]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[108]:


X = dropdata[["Weeks","Bolivia"]]
#Visualise data points
plt.scatter(X["Weeks"],X["Bolivia"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[109]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Bolivia"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Bolivia"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[110]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Bolivia"]-row_d["Bolivia"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Bolivia","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Bolivia'] - Centroids['Bolivia']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Bolivia","Weeks"]]


# In[111]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Bolivia"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Bolivia"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Peru 

# In[112]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[113]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[114]:


data.describe()


# In[115]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[116]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[117]:


kmeans.inertia_


# In[118]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[119]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[120]:


X = dropdata[["Weeks","Peru"]]
#Visualise data points
plt.scatter(X["Weeks"],X["Peru"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[121]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Peru"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Peru"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[122]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Peru"]-row_d["Peru"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Peru","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Peru'] - Centroids['Peru']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Peru","Weeks"]]


# In[123]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Peru"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Peru"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Colombia

# In[124]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[125]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[126]:


data.describe()


# In[127]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[128]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=5, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[129]:


kmeans.inertia_


# In[130]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,7))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[131]:


model = KMeans(n_clusters=5)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[132]:


X = dropdata[["Weeks","Colombia"]]
#Visualise data points
plt.scatter(X["Weeks"],X["Colombia"],c = 'black')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[133]:


K=5

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Colombia"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Colombia"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[134]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Colombia"]-row_d["Colombia"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Colombia","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Colombia'] - Centroids['Colombia']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Colombia","Weeks"]]


# In[135]:


color=['blue','green','cyan','black','yellow']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Colombia"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Colombia"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Venezuela

# In[136]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[137]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[138]:


data.describe()


# In[139]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[140]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=4, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[141]:


kmeans.inertia_


# In[142]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[143]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[144]:


X = dropdata[["Weeks","Venezuela"]]
#Visualise data points
plt.scatter(X["Weeks"],X["Venezuela"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[145]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Venezuela"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Venezuela"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[146]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Venezuela"]-row_d["Venezuela"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Venezuela","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Venezuela'] - Centroids['Venezuela']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Venezuela","Weeks"]]


# In[147]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Venezuela"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Venezuela"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Guyana

# In[148]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[149]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Suriname',axis=1,inplace=True)

data.head()


# In[150]:


data.describe()


# In[151]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[152]:


kmeans = KMeans(n_clusters=4, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[153]:


kmeans.inertia_


# In[154]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[155]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[156]:


X = dropdata[["Weeks","Guyana"]]
#Visualise data points
plt.figure(figsize=(10, 8))
plt.scatter(X["Weeks"],X["Guyana"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[157]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Guyana"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Guyana"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[158]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Guyana"]-row_d["Guyana"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Guyana","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Guyana'] - Centroids['Guyana']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Guyana","Weeks"]]


# In[159]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Guyana"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Guyana"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# # Suriname

# In[160]:


data = pd.read_csv('Neighbouring.csv')
data.head()


# In[161]:


dropdata = data

dropdata.drop('Uruguay',axis=1,inplace=True)
dropdata.drop('Argentina',axis=1,inplace=True)
dropdata.drop('Paraguay',axis=1,inplace=True)
dropdata.drop('Bolivia',axis=1,inplace=True)
dropdata.drop('Peru',axis=1,inplace=True)
dropdata.drop('Colombia',axis=1,inplace=True)
dropdata.drop('Venezuela',axis=1,inplace=True)
dropdata.drop('Guyana',axis=1,inplace=True)

data.head()


# In[162]:


data.describe()


# In[163]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[164]:


kmeans = KMeans(n_clusters=4, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[165]:


kmeans.inertia_


# In[166]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[167]:


model = KMeans(n_clusters=4)

# Note I'm scaling the data to normalize it! Important for good results.

model = model.fit(scale(data))

# print(scale(data))
# print(data)

# We can look at the clusters each data point was assigned to
print(model.labels_)


# In[168]:


X = dropdata[["Weeks","Suriname"]]
#Visualise data points
plt.figure(figsize=(10, 8))
plt.scatter(X["Weeks"],X["Suriname"],c = model.labels_.astype(np.int64))
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[169]:


K=4

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Weeks"],X["Suriname"],c= 'black')
plt.scatter(Centroids["Weeks"],Centroids["Suriname"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[170]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Weeks"]-row_d["Weeks"])**2
            d2=(row_c["Suriname"]-row_d["Suriname"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Suriname","Weeks"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Suriname'] - Centroids['Suriname']).sum() + (Centroids_new['Weeks'] - Centroids['Weeks']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Suriname","Weeks"]]


# In[171]:


color=['blue','green','cyan','black']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Weeks"],data["Suriname"],c=color[k])
plt.scatter(Centroids["Weeks"],Centroids["Suriname"],c='red')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.show()


# In[ ]:




