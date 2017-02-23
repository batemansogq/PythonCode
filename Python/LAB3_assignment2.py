import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
wheat = pd.read_csv("C:/Users/mckinns/Documents/GitHub/DAT210x/Module3/Datasets/wheat.data", 
                              index_col=0, header=0)
df_wh = pd.DataFrame(wheat)


# TODO: Create a 2d scatter plot that graphs the
# area and perimeter features
wheat.plot.scatter(x='area', y='perimeter')
plt.suptitle('area vs perimeter ')
plt.xlabel('area')
plt.ylabel('perimeter')

# groove and asymmetry features
wheat.plot.scatter(x='groove', y='asymmetry')
plt.suptitle('groove vs asymmetry')
plt.xlabel('groove')
plt.ylabel('asymmetry')

# compactness and width features
wheat.plot.scatter(x='compactness', y='width')
plt.suptitle('compactness vs width')
plt.xlabel('compactness')
plt.ylabel('width')

# BONUS TODO:
# After completing the above, go ahead and run your program
# Check out the results, and see what happens when you add
# in the optional display parameter marker with values of
# either '^', '.', or 'o'.


plt.show()


