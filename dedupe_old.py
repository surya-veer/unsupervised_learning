# $ 0 u l $ h i f t 3 r

# IMPORTING DATA HANDLING LIBRARIES
import numpy as np
import sys
import pandas as pd
pd.set_option("display.height",2000)
pd.set_option("display.max_rows",2000)
pd.set_option("display.max_columns",2000)
pd.set_option("display.width",2000)
pd.set_option("display.max_colwidth",-1)

import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use("ggplot")

import seaborn as sns
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")


# LOAD DATA

train = pd.read_csv("/home/maniac/Desktop/Kaggle/Deduplication/train.csv")
final = pd.DataFrame({"ln":train["ln"],"dob":train["dob"],"gn":train["gn"],"fn":train["fn"]},columns=["fn","ln","dob","gn"])

# CHECK DATA

# print(train.head(20))
# print(train.shape)
# print(train.info())
# print(train.describe())

# CHECK NULL FIELDS

# print(train.isnull().sum())

# PREPROCESSING DATA

train["gn"] = train["gn"].map({"M":1,"F":0})
train["gn"] = train["gn"].astype(int)
train["dob"] = train["dob"].apply(lambda x: x.replace("/",""))
train["dob"] = train["dob"].apply(lambda x: x[1:] if x[0] == "0" else x[0:])
train["dob"] = train["dob"].astype(float)
from sklearn.preprocessing import StandardScaler
train["dob"] = StandardScaler().fit_transform(train["dob"].values.reshape(-1,1))
# print(train["dob"].head())

# print(train.info())
# print(train["dob"].head())
#
train["ln"] = train["ln"].apply(lambda x: x.split())
train["ln"] = train["ln"].apply(lambda x: "".join(x))
train["ln"] = train["ln"].apply(lambda x: x.lower())
train["fn"] = train["fn"].apply(lambda x: x.split())
train["fn"] = train["fn"].apply(lambda x: "".join(x))
train["fn"] = train["fn"].apply(lambda x: x.lower())
#
# # print(train["ln"])
import codecs
def convert(x) :
    return str(int(codecs.encode(x.encode(),"hex_codec"),base=16)/sys.maxsize)

train["ln"] = train["ln"].apply(lambda x:convert(x))
train["ln"] = train["ln"].astype(float)
train["fn"] = train["fn"].apply(lambda x:convert(x))
train["fn"] = train["fn"].astype(float)
train["ln"] = StandardScaler().fit_transform(train["ln"].values.reshape(-1,1))
train["fn"] = StandardScaler().fit_transform(train["fn"].values.reshape(-1,1))
# print(train.dtypes)
# print(train["fn"].head())
# print(train.head(20))
# print(train.info())

# MODELLING

from sklearn.cluster import KMeans

n_cluster = len(train.index)
kmn = KMeans(n_clusters=n_cluster,max_iter=300,precompute_distances=True,n_jobs=-1)
kmn.fit(train)
target = kmn.predict(train)
final["Target"] = target
# print(final)
print(len(final["Target"].unique()))
# final.drop_duplicates(subset="Target",inplace=True)
# print(final.head())
# print(final.shape)
print(final)
# final.to_csv("/home/maniac/Desktop/Kaggle/Deduplication/final.csv")
