import pandas as pd



# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
# pull the data
df_ht = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2'
                     ,skiprows=1, header=0)
# extract the table from the list
df_ht1 = pd.DataFrame(df_ht[0])


# TODO: Rename the columns so that they are similar to the
# column definitions provided to you on the website.
# Be careful and don't accidentially use any names twice.
#
df_ht1.columns = ['RK', 'PLAYER' , 'TEAM' , 'GP' , 'G' , 'A' , 'PTS' , '+/-' , 'PIM' , 'PTS/G', 	
              'SOG' , 'PCT' , 'GWG' , 'PP G' , 'PP A' , 'SH G' , 'SH A']

# TODO: Get rid of any row that has at least 4 NANs in it,
# e.g. that do not contain player points statistics
#
df_ht1 = df_ht1.dropna(axis=0, thresh=4)

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
df_ht1 = df_ht1[(df_ht1.RK!='RK')]


# TODO: Get rid of the 'RK' column
#
df_ht1 = df_ht1.drop(labels=['RK'], axis=1)
print df_ht1

# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
df_ht1.reset_index()
#drop old index
df_ht1 = df_ht1.drop(labels=['index'], axis=1)

# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
#

df_ht1.GP = pd.to_numeric(df_ht1.GP, errors='coerce')
df_ht1.G = pd.to_numeric(df_ht1.G, errors='coerce')
df_ht1.A = pd.to_numeric(df_ht1.A, errors='coerce')
df_ht1.PTS = pd.to_numeric(df_ht1.PTS, errors='coerce')
df_ht1['+/-'] = pd.to_numeric(df_ht1['+/-'], errors='coerce')
df_ht1.PIM = pd.to_numeric(df_ht1.PIM, errors='coerce')
df_ht1['PTS/G'] = pd.to_numeric(df_ht1['PTS/G'], errors='coerce')
df_ht1.SOG = pd.to_numeric(df_ht1.SOG, errors='coerce')
df_ht1.PCT = pd.to_numeric(df_ht1.PCT, errors='coerce')
df_ht1.GWG = pd.to_numeric(df_ht1.GWG, errors='coerce')
df_ht1['PP G'] = pd.to_numeric(df_ht1['PP G'], errors='coerce')
df_ht1['PP A'] = pd.to_numeric(df_ht1['PP A'], errors='coerce')
df_ht1['SH G'] = pd.to_numeric(df_ht1['SH G'], errors='coerce')
df_ht1['SH A'] = pd.to_numeric(df_ht1['SH A'], errors='coerce')

df_ht1.dtypes

# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.
#
# .. your code here ..
# how many rows
print len(df_ht1)
# How many unique PCT values exist in the table?
len(df_ht1.PCT.unique())
#What is the value you get by adding the GP values at indices 15 and 16 of this table?
print df_ht1
print df_ht1.loc[15:16, 'GP']
print sum(df_ht1.loc[15:16, 'GP'])