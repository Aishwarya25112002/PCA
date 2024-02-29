# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:16:14 2023

@author: hp

#DataSet - heart disease.csv
         
#BUSINESS UNDERSTANDING

Maximise:-Daily exercise which lead to achieve
max. heart rate,Blood Circulation and good health

Minimise:-Minimise trestbps,cholesterol,thal. 
Reduce the no. Of Heartdiseases patients.

Business Constraints:-To maintain the level
of chol(cholesterol),if it is beyond
its limit consider risky

Exploratory Data Analysis(EDA)

# DATA DICTIONARY-

Name of             Description                                             Type                                         Relevance   
Feature
                                                                    Qualitative(Nominal,Ordinal)                      [Relevant,Irrelevant]
                                                                    Quantitative(Discrete,Continuous)             (i.e presents col.provide useful info or not???
                                                                                                                   age       Age of the patient                                       Continuous(Quantitative)                        Relevant           sex       Gender of the patient (1 = male,0 = female).             Nominal (Qualitative)                           Relevant
                                                                                                                   cp        Chest pain type.                                         Ordinal (Qualitative)                           Relevant
trestbps  Resting blood pressure (in mm Hg).                       Continuous(Quantitative)                        Relevant
chol      Serum cholesterol level (in mg/dL).                      Quantitative (Continuous)                       Relevant
fbs       Fasting blood sugar level. If fbs>120                    Nominal (Qualitative)                           Relevant
         it is represented as 1. Otherwise, it is 0.              
restecg   Resting electrocardiographic results.                    Nominal (Qualitative)                           Relevant
thalach   Maximum heart rate achieved.                             Quantitative (Continuous)                       Relevant 
exang     Exercise-induced angina (1 = yes,0 = no).                Nominal (Qualitative)                           Relevant
oldpeak   ST depression induced by exercise relative to rest.      Quantitative (Continuous)                       Relevant
slope     The slope of the peak exercise ST segment.               ordinal (Qualitative)                           Relevant
ca        Number of major vessels                                  Quantitative (Discrete)                         Relevant 
thal      Thalassemia. It is a blood disorder.                     Nominal (Qualitative)                           Relevant
target    The presence of heart disease(1=presence,0=absence).     Nominal (Qualitative                            Relevant

"""
# import the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df1= pd.read_csv("C:/8-PCA/heart disease.csv")
df1

#########  DATA PRE-PROCESSING ############

#Data Cleaning and Feature Engineering
df1.columns
'''Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
      dtype='object')
                                                                            
 '''                               
#(total 14 columns are there.)

######
df1.shape
"""(rows-303,col-14)"""

df1.info() # show the all information
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trestbps  303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalach   303 non-null    int64  
 8   exang     303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64  
 11  ca        303 non-null    int64  
 12  thal      303 non-null    int64  
 13  target    303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
'''
#########Exploratory Data Analysis #########

#Univarient Analysis
"""PROCESS:-
Check Data Types
Display the First Few Rows
Check for Duplicates
Checking for duplicates 
Handle Missing Values
Check Outliers"""

#check the data type of columns
df1.dtypes

"""all col. have int64 data type ,only oldpeak col. have data type float."""
## Summary

summary = df1.describe()
print(summary)

################################
#Display the first few rows of the dataset
y=df1.head()
"""we display intial 5 rows data"""

############################
#check whether null value is present or not
df1.isnull().sum()
"""there is not a single null value present"""
################################3

#(No need to write this code bcz we don't have any missing value)
# Handle missing values
m=df1.fillna(df1.mean(), inplace=True)
m
"""
as we check there is no missing value is present in this data
so need to write this code
Fills missing values with the mean of each column.
"""
######################################################
# Check for duplicates
df1.duplicated().sum()
#o/p 1
""" 1 duplicate row is avaliable
here in this dataset duplicate values are present.
That duplicate value introduce some error in our model 
and also we get less accuracy.
we need remove that duplicate data,To get more accuracy

before we have (rows=303 & col=14)
"""
#####################################################

#Remove duplicate values
d=df1.drop_duplicates()
d
"""Removed duplicate row
now data have 302 rows and 14 column"""

####################################################

#Check Whether Outlier present or not
"""
affect the mean value(inaccurate mean value) of the
data but have little effect on the median or mode.
Outliers can have a big impact on
analyses test if they are inaccurate. """

#various method we have to check outliers
#    scatter plot
#    Box plot
#    Z-score
#    IQR (Interquartile Range Q1=25%,Q2=50%median,Q3=75%)

# Identify & handel outliers (using IQR method)
Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
#removing outlier from data
no_outliers = df1[~((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
"""Here we remove the outlier which is greater than 1.5 and
less than 1.5"""
#we know 
#If data value  < Q1 - 1.5*IQR  =then outlier
#If data value  > Q3 + 1.5*IQR  =then outlier
no_outliers.shape

"""(228, 14)
some outlier rows is removed from data"""


####################################################3
#Technique-2 for finding outlier

#               Bivariate Analysis
#       Process
"""
Correlation Analysis:[ use df1.corr()]
Box Plot
Standardization
One-Hot Encoding
"""

#1] Correlation Analysis
# Correlation Analysis
correlation_matrix = df1.corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

##################################################

#2]Box Plot

sns.boxplot(df1)
"""trestbps,chol,thalach have outliers"""
#(automatic take number column which have variation in no.)

#same in above code
# box plot numerical columns
#plt.figure(figsize=(12, 8))
#sns.boxplot(data=df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
#plt.title('Box Plots for Numerical Columns')
#plt.show()


###############################################################################
       # Standardization numerical
"""-To bring all features to a similar scale
   -To make effective Regularization techniques, such as L1 or L2 regularization
   
   L1=Lasso Regularization
   term is the absolute sum of the model's coefficients. 
   
   L2=Ridge Regularization
   term is the squared sum of the model's coefficients
"""

from sklearn.preprocessing import StandardScaler

#columns_to_standardize is a list of column names to standardize
columns_to_standardize = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
df1[columns_to_standardize] = scaler.fit_transform(df1[columns_to_standardize])

###################################################################################

#3] Standardization

#Another Way for standardization  
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df1[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(df1[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

# Convert data types
df1['sex'] = df1['sex'].astype('category')

############################################################
"""
#There is no need of normalisation
#It's only practice purpose
#Normalization
from sklearn.preprocessing import MinMaxScaler


# Assuming 'data' is your DataFrame and 'columns_to_standardize' is a list of column names to standardize
columns_to_normalize = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Initialize the StandardScaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
df1[columns_to_normalize] = scaler.fit_transform(df1[columns_to_normalize])
"""

#####################################################
#4. ONE HOT ENCODING
#(Feature Engineering)
#One hot Encoding is used :-
    #1] Algorithms Requiring Numerical Input
    #In ML algorithms, such as linear models and
    #distance-based models, require numerical input
   #2]Preventing Multicollinearity

# Identify categorical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns=categorical_columns, drop_first=True)

#5 .Model Building

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df1)

# PCA analysis
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

#################################
#Clustering -
# Select relevant features
X = df1.drop('target', axis=1)

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df1['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the results using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

########################################
#Association rules

#Apriori Algorithms******

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Select relevant features
X = df1.drop('target', axis=1)

# Convert the data to one-hot encoding
X_encoded = pd.get_dummies(X)

# Apply Apriori algorithm
frequent_itemsets = apriori(X_encoded, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the association rules
print("Association Rules:")
print(rules)
