import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
df_sr = pd.read_csv('C://Users/mckinns/Documents/GitHub/DAT210x/Module2/Datasets/servo.data', 
                     sep=',', names=['motor', 'screw', 'pgain', 'vgain', 'class'])

print df_sr.head(5)


# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
print df_sr[(df_sr.vgain==5)].head(3)
print len(df_sr[(df_sr.vgain==5)])

# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
print df_sr[(df_sr.motor=='E') & (df_sr.screw=='E')].head(3)
print len(df_sr[(df_sr.motor=='E') & (df_sr.screw=='E')])




# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#

print df_sr[(df_sr.pgain==4)].describe()



# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!



