# %% 
from loader import PROJECT_ROOT
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
#from loader import PROJECT_ROOT
#from functions import categorical_sanity_check
#from functions import validate_dtypes
#from functions import missing_values_report
#from functions import plot_distribution
#from functions import plot_boxplot
#import functions
# %%

#load data
import tarfile
import urllib.error
import logging
import requests 
from  tqdm import tqdm


# %%
def load_housing_data():
  tarball_path = Path("datasets/housing.tgz")
  if not tarball_path.is_file():
    Path("datasets").mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
      housing_tarball.extractall(path="datasets", filter="data")
  return pd.read_csv(Path("datasets/housing/housing.csv"))


housing_full = load_housing_data()
housing_full
housing_full['ocean_proximity'].value_counts()


# Histogram of numeric features for understanding and spread
housing_full.hist(bins=50, figsize=(15, 8))
plt.show()

"""
Create and split your test data set right now 
This helps to fight data leakage
"""
import numpy as np

def shuffle_and_split_data(data, test_ratio, rng):
  shuffled_indices = rng.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

"""
lets call the function
"""
rng = np.random.default_rng() # default random number generator
train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
len(train_set)

len(test_set)

"""
this function is good but the data will be sliding when recalled,
its better to lock it with the index and build so that it further 
splits in the percentage you have set for it even when new data
comes in, so we give the df a datframe and split
"""
from zlib import crc32
def is_id_in_test_set(identifier, test_ratio):
  return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
  return data.loc[~in_test_set], data.loc[in_test_set]

# using the row index as the ID
housing_with_id = housing_full.reset_index() #adds an 'index' column and also makes it a dframe
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# using log and lat combined  as ID cos  they are unique
# but be careful of bias due to location concentrate
housing_with_id["id"] = (housing_full["longitude"] * 1000
                         + housing_full["latitude"])
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size =0.2,
                                       randaom_state=42)
"""
# The last 2 train and test split has been random, 
# this might neglect some data sets not appearing due to low number
# hence we opt for stratified sampling
# we are implementing straified sampling on the income_category
# because it is shown to have a direct impact on the housing average so
# it will be important to make sure all features are grouped capturing 
# all section of the average income.it will help us
"""
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1,2,3,4,5])

# now lets plot the income categories
cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

# next we splitting using the income strat as a basis for split
housing_full


housing_with_id