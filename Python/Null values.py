import pandas as pd

###############################
# deal with missing data
###############################

df = pd.read_csv('demo.csv', sep='\t')
print df

#test for missing data
print df.Unemployed.isnull()
print df.noynull()

#fill all NaN's will a scalar
print df.fillna(0)

#fill foward with previous value
print df.fillna(method='ffill', limit=1)


#interpolate with polynomial
print df.interpolate(method='polynomial', order=2)

#drop all rows with NaN's
df.dropna(axis=0)
#drop all columns with NaN
df.dropna(axis=1)
# drop some columes
df = df.drop(labels=['Features', 'To', 'Delete'], axis=1)

# Drop any row that has at least 4 NON-NaNs within it:
df = df.dropna(axis=0, thresh=4)


# drop duplicates, where teh unq key is a subset of features
print df.drop_duplicates(subset=['Major category', 'Total']).reset_index()



