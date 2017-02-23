import pandas as pd

csv_dataframe  = pd.read_csv('E:\\R\\Football\\2010.csv', sep=',')
print csv_dataframe.head(5)
print csv_dataframe.columns
print csv_dataframe.describe()
print csv_dataframe.index
print csv_dataframe.dtypes

##########################
# 2. Data And Features > Lecture: Manipulating Data > Slicin'
############################

csv_df  = pd.read_csv('C://Users/mckinns/Documents/GitHub/DAT210x/Module2/Datasets\direct_marketing.csv',
                      sep=',') 

print csv_df.dtypes
# logical subset, limited view
csv_df[(csv_df.recency < 4) & (csv_df.newbie==0)].head(3)

##################################
# 2. Data And Features > Lecture: Feature Representation
###################################

ordered_satisfaction = ['Very Unhappy', 'Unhappy', 'Neutral', 'Happy', 'Very Happy']
df = pd.DataFrame({'satisfaction':['Mad', 'Happy', 'Unhappy', 'Neutral']})
df.satisfaction = df.satisfaction.astype("category",
  ordered=True,
  categories=ordered_satisfaction
).cat.codes

df

# default
df = pd.DataFrame({'vertebrates':[
  'Bird',
  'Bird',
  'Mammal',
  'Fish',
  'Amphibian',
  'Reptile',
  'Mammal', ]})

# Method 1)
df['vertebrates'] = df.vertebrates.astype("category").cat.codes
df

# Method 2 explode out to columns
df = pd.get_dummies(df,columns=['vertebrates'])
df