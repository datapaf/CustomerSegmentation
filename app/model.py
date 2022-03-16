# import libraries
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib_inline
#%matplotlib inline

# set random seed
import random
random.seed(335)

# read excel file
print('-> Reading Data...')
df = pd.read_excel("Online Retail.xlsx")
print('-> Reading Done!')

print('-> Data Preprocessing...')

# Remove register without CustomerID
df = df[~(df.CustomerID.isnull())]

# Remove negative or return transactions
df = df[~(df.Quantity<0)]
df = df[df.UnitPrice>0]

# transformation to the necessary datatypes
df.InvoiceDate = pd.to_datetime(df.InvoiceDate)
df.CustomerID = df.CustomerID.astype('Int64')

# Amount
df['amount'] = df.Quantity*df.UnitPrice

# Days since Last Purchase
import datetime
refrence_date = df.InvoiceDate.max() + datetime.timedelta(days = 1)
df['days_since_last_purchase'] = (refrence_date - df.InvoiceDate).astype('timedelta64[D]')

# Recency
customer_history_df = df[['CustomerID', 'days_since_last_purchase']].groupby("CustomerID").min().reset_index()
customer_history_df.rename(columns={'days_since_last_purchase':'recency'}, inplace=True)

# Frequency
customer_freq = (df[['CustomerID', 'InvoiceNo']].groupby(["CustomerID", 'InvoiceNo']).count().reset_index()).groupby(["CustomerID"]).count().reset_index()
customer_freq.rename(columns={'InvoiceNo':'frequency'},inplace=True)
customer_history_df = customer_history_df.merge(customer_freq)

# Monetary Value
customer_monetary_val = df[['CustomerID', 'amount']].groupby("CustomerID").sum().reset_index()
customer_history_df = customer_history_df.merge(customer_monetary_val)

import math
from sklearn import preprocessing

# transform the variables on the log scale 
# (solves the problem with a huge range of values)
customer_history_df['recency_log'] = customer_history_df['recency'].apply(math.log)
customer_history_df['frequency_log'] = customer_history_df['frequency'].apply(math.log)
customer_history_df['amount_log'] = customer_history_df['amount'].apply(math.log)

# standardization (necessary for K-means)
feature_vector = ['amount_log', 'recency_log','frequency_log']
X_subset = customer_history_df[feature_vector] #.as_matrix()
scaler = preprocessing.StandardScaler().fit(X_subset)
X_scaled = scaler.transform(X_subset)

print('-> Preprocessing Done!')

# build and train the model with the optimal parameters
from sklearn.cluster import KMeans

print('-> Preparing Model...')

clusterer = KMeans(
    n_clusters=7,
    init='k-means++', 
    random_state=101
)

clusterer.fit(X_scaled)

print('-> Done')