import pandas as pd
import numpy as np


#
# TODO:
# Load up the dataset, setting correct header labels.
#
df_cn = pd.read_csv('C://Users/mckinns/Documents/GitHub/DAT210x/Module2/Datasets/census.data', 
                    sep=',', names = ['education', 'age', 
                    'capital-gain', 'race', 'capital-loss', 
                    'hours-per-week', 'sex', 'classification'], na_values=['?'])

print df_cn

#
# TODO:
# Use basic pandas commands to look through the dataset... get a
# feel for it before proceeding! Do the data-types of each column
# reflect the values you see when you look through the data using
# a text editor / spread sheet program? If you see 'object' where
# you expect to see 'int32' / 'float64', that is a good indicator
# that there is probably a string or missing value in a column.
# use `your_data_frame['your_column'].unique()` to see the unique
# values of each column and identify the rogue values. If these
# should be represented as nans, you can convert them using
# na_values when loading the dataframe.
#

df_cn.dtypes
df_cn['education'].unique()
df_cn['race'].unique()
df_cn['sex'].unique()
df_cn['classification'].unique()
df_cn.age.unique()

df_cn = pd.DataFrame(df_cn)

df_cn.dtypes
df_cn.age.value_counts()


#
# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal and nominal
# types using the methods discussed in the chapter.
#
# Be careful! Some features can be represented as either categorical
# or continuous (numerical). If you ever get confused, think to yourself
# what makes more sense generally---to represent such features with a
# continuous numeric type... or a series of categories?
# 
df_cn['classification'] = df_cn['classification'].astype("category",
  ordered=True,
  categories=['<=50K', '>50K']
).cat.codes

df_cn.head(15)
#explode features out
df_cn = pd.get_dummies(df_cn,columns=['sex'])
df_cn = pd.get_dummies(df_cn,columns=['race'])
#
# TODO:
# Print out your dataframe
#
# .. your code here ..


