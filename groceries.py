# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:02:29 2023

@author: hp
   
        DataSet-Groceries

#Business Objective:
Minimize -inventory costs by optimizing stock 
          levels for less common combinations

Maximize-revenue through targeted promotions for
         frequently paired items.

Constraints:Ensure that the quantity of each grocery 
           item is non-negative.Guarantee that each
           transaction has a unique identifier
           
#Data Dictionary

Name Of 
Features                   Descrioption                                    Relevent
citrus fruit         It is a type of refers to fruits that                 Relevent
                         belong to the citrus genus         

semi-finished bread   partially baked product Bread                        Relevent
margarine             Margarine is a spread made from vegetable oils       Irrelevent
ready soups           packed redemade soups packet                         Irrelevent

here margarine and ready soups is irrlevent
bcz a human can leave by cunsuming only fruits and bread
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

#load the dataset 
data=pd.read_csv("C:/8-PCA/groceries.csv")
data

#################################
#Explore Data
data.columns
#It show the column in dataset

# Drop columns with all NaN values
data = data.dropna(axis=1, how='all')

# Drop rows with any NaN values
data = data.dropna(axis=0, how='any')

# Fill NaN values with a specific value or strategy (e.g., forward fill)
data = data.fillna(method='ffill')
'''Perform any additional data cleaning or feature
 engineering as needed
'''

# Display the cleaned dataset
print(data)

data.shape
#it show the number of rows and column
#(1,32)

data.columns  #that show column names

data.info #that shows all information 

data.describe() #provides descriptive statistics

# Display a summary of the dataset
summary = data.describe()
print("Summary of the Dataset:")
print(summary)

# Univariate Analysis - Count of unique values in each column
univariate_counts = data.nunique()
print(univariate_counts)

# Bivariate Analysis - Visualization of associations between 'citrus fruit' and 'semi-finished bread'
plt.figure(figsize=(10, 6))
sns.countplot(x='citrus fruit', hue='semi-finished bread', data=data)
plt.title('Association between Citrus Fruit and Semi-Finished Bread')
plt.xlabel('Citrus Fruit')
plt.ylabel('Count')
plt.show()

# Assume 'citrus fruit' and 'semi-finished bread' are relevant columns for analysis
selected_columns = ['citrus fruit', 'semi-finished bread']
selected_data = data[selected_columns]

# Encode categorical variables if needed

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# Build a PCA model
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Plot the explained variance ratio
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio')
plt.show()

# Choose the number of components based on the plot
n_components = 2  # Adjust based on your analysis

# Apply PCA with the selected number of components
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)
#################################
#     Clustering
# Perform clustering
kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters based on your analysis
clusters = kmeans.fit_predict(pca_result)

# Visualize the clusters
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
plt.title('PCA with Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Include the cluster labels in the original dataset
data['Cluster'] = clusters

# Display the dataset with cluster labels
print(data)

####################################
#Association rules

#Apriori Algorithms

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Select relevant features
X = data.drop('target', axis=1)

# Convert the data to one-hot encoding
X_encoded = pd.get_dummies(X)

# Apply Apriori algorithm
frequent_itemsets = apriori(X_encoded, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the association rules
print("Association Rules:")
print(rules)
