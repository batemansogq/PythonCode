from pandas.tools.plotting import andrews_curves

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')

wheat = pd.read_csv("C:/Users/mckinns/Documents/GitHub/DAT210x/Module3/Datasets/wheat.data", 
                              index_col=0, header=0)

# Also get rid of the 'area' and 'perimeter' features
df_lim = wheat.drop([ 'area','perimeter'], axis=1)

# andrew curves
plt.figure()
andrews_curves(df_lim, 'wheat_type', alpha=0.4)
plt.show()

plt.figure()
andrews_curves(wheat, 'wheat_type', alpha=0.4)
plt.show()