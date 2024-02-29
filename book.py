# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:59:37 2023

@author: hp
         Dataset-Book
"""
'''
Business Objective:
    
maximize: Identify and promote book combinations that are 
frequently purchased together to increase cross-selling opportunities.

Minimize: Increase sales and revenue by promoting popular book categories 

Constraints: The business needs to address online competition.
Strategies should include both online and offline components to capture a broader market.
'''

'''
Data dictionary:

Nominal Data:

'ChildBks': Children's books category.
'YouthBks': Youth books category.
'CookBks': Cookbooks category.
'RefBks': Reference books category.
'ArtBks': Art books category.
'GeogBks': Geography books category.
'ItalCook': Italian Cookbooks category.
'ItalAtlas': Italian Atlases category.
'ItalArt': Italian Art books category.
'Florence': Possibly a location or specific book related to Florence.

Ordinal Data:

'DoItYBks': Do-it-yourself books category.
'''
#import the data
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset

df=pd.read_csv('book.csv')
df

####################################
df.columns
#Show the all column in data
'''
['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
       'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence']
'''
###################################
df.shape
#(2000,11)
# the dataset contain 2000 rows and 11 columns

###################################
df.dtypes

'''
ChildBks     int64
YouthBks     int64
CookBks      int64
DoItYBks     int64
RefBks       int64
ArtBks       int64
GeogBks      int64
ItalCook     int64
ItalAtlas    int64
ItalArt      int64
Florence     int64

The datatype is of numeric type there is no need of encoding
'''
######################################
a=pd.isnull(df)
a.sum()
'''
ChildBks     0
YouthBks     0
CookBks      0
DoItYBks     0
RefBks       0
ArtBks       0
GeogBks      0
ItalCook     0
ItalAtlas    0
ItalArt      0
Florence     0
dtype: int64

As there is no null value in the dataset
'''
#####################################
q=df.value_counts()
####################################
# Five Number Summary
v=df.describe()
# The mean value is near to zero and also the standard deviation is a;dp
# near to zero and the meadian value for the all datapoints is zero
df.info()
'''
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ChildBks   2000 non-null   int64
 1   YouthBks   2000 non-null   int64
 2   CookBks    2000 non-null   int64
 3   DoItYBks   2000 non-null   int64
 4   RefBks     2000 non-null   int64
 5   ArtBks     2000 non-null   int64
 6   GeogBks    2000 non-null   int64
 7   ItalCook   2000 non-null   int64
 8   ItalAtlas  2000 non-null   int64
 9   ItalArt    2000 non-null   int64
 10  Florence   2000 non-null   int64
dtypes: int64(11)
'''
# This will give us the informationn about all the points

####################################
#           Visualization of Data

# 1. Check for the outlier

sns.boxplot(df,x='ChildBks')
# No outlier 
sns.boxplot(df,x='YouthBks')
#There is one outlier 
sns.boxplot(df,x='CookBks')
# No Outlier
sns.boxplot(df,x='RefBks')
# There is one outlier
sns.boxplot(df)
# Observe that some columns contain  the outlier so we have to normalize it

#2. Pairplot
sns.pairplot(df)
# No Datapoints are corelated as the all the datapoints are in scatter form 

#3. Heatmap
corr=df.corr()
sns.heatmap(corr)
# The diagonal color of the heatmap is same as the datapoints folllow some pattern
# so we can use this data for the model building
############################################
#           Normalization

#The data is numeric one so we have to perform normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(v)
df_norm

b=df_norm.describe()

sns.boxplot(df_norm)
# No Outlier is remaining
# The all the quantile points are converted in the rande of 0-1
############################################
# Model Building
#           Association Rules
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv('book.csv')
data

# All the data is in properly separated form so no need to apply the encoding techique
# as it is already is in the form of numeric one

from collections import Counter
item_frequencies=Counter(data)

# Apriori algorithm
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# This generate association rule for columns
# comprises of antescends,consequences

rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

# Visualize the rules
import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph from the rules
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')

# Draw the graph
fig, ax = plt.subplots(figsize=(14, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.title("Association Rules Network", fontsize=15)
plt.show()

#####################################
#          Clustering

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming you have selected relevant features for clustering, e.g., 'rating' and 'price'
selected_features = ['rating', 'price']
X = df[selected_features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you may need to adjust this based on your data)
num_clusters = 3

# Apply k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters (assuming 2D features)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis', edgecolor='k')
plt.title('K-Means Clustering of Books Data')
plt.xlabel('Rating (Standardized)')
plt.ylabel('Price (Standardized)')
plt.show()
###################################################
#PCA 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming you have a book dataset in a CSV file called 'books.csv'
# Load the dataset
df = pd.read_csv('books.csv')

# Assuming you have selected relevant features for PCA, e.g., 'feature1' and 'feature2'
selected_features = ['feature1', 'feature2']
X = df[selected_features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
df_pca = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])

# Visualize the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Visualize the principal components
plt.scatter(df_pca['Principal Component 1'], df_pca['Principal Component 2'])
plt.title('PCA of Books Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# the benefits/impact of the solution 
# By identifying books that are frequently purchased together,
# the bookstore can create curated bundles or recommendations, enhancing the overall 
# shopping experience for customers.
# By using this association rule we can stratergically placed the books together to encourage
# the customer to purchased more items which will help to increased the overall revenue

