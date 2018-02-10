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

# print(train)
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
train["dob"] = train["dob"].astype(int)
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
# print(train.dtypes)
# print(train["fn"].head())
# print(train.head(20))
# print(train.info())

# MODELLING

from sklearn.cluster import KMeans

n_cluster = len(train.index)
kmn = KMeans(n_clusters=n_cluster)
kmn.fit(train)
target = kmn.predict(train)
final["Target"] = target
# print(final)
print(len(final["Target"].unique()))
# final.to_csv("/home/maniac/Desktop/Kaggle/Deduplication/final.csv")