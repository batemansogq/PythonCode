import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset
#
df_csv = pd.read_csv('C://Users/mckinns/Documents/GitHub/DAT210x/Module2/Datasets/tutorial.csv', sep=',')

print df_csv.head(5)

# TODO: Print the results of the .describe() method
#
df_csv.describe()



# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
df_csv.loc[2:4,'col3']

