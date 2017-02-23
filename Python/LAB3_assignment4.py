import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

wheat = pd.read_csv("C:/Users/mckinns/Documents/GitHub/DAT210x/Module3/Datasets/wheat.data", 
                              index_col=0, header=0)

# Also get rid of the 'area' and 'perimeter' features
df_lim = wheat.drop([ 'area','perimeter'], axis=1)

# the 'wheat_type' feature. display parameter alpha to 0.4
plt.figure()
parallel_coordinates(df_lim, 'wheat_type', alpha=0.4)
plt.show()


