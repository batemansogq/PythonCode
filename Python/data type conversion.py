import pandas as pd

df.dtypes

#Date        object
#Name        object
#Gender      object
#Height      object
#Weight      object
#Age         object
#Job         object

#If your data types don't look the way you expected them, explicitly convert them to the desired type using the .to_datetime(), .to_numeric(), and .to_timedelta() methods:

df.Date = pd.to_datetime(df.Date, errors='coerce')
df.Height = pd.to_numeric(df.Height, errors='coerce')
df.Weight = pd.to_numeric(df.Weight, errors='coerce')
df.Age = pd.to_numeric(df.Age, errors='coerce')
df.dtypes

#Date        datetime64
#Name        object
#Gender      object
#Height      float64
#Weight      float64
#Age         int64
#Job         object

# display unq values
df.Age.unique()

#array([7, 33, 27, 40, 22], dtype=int64)


# how many values of each?
df.Age.value_counts()

#7      1
#22     5
#27     1
#33     2
#40     2
#dtype: int64