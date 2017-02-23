import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

wheat = pd.read_csv("C:/Users/mckinns/Documents/GitHub/DAT210x/Module3/Datasets/wheat.data", 
                              index_col=0, header=0)

df_wh = pd.DataFrame(wheat)
s1 = df_wh[['perimeter', 'area']]
s2 = df_wh[['groove','asymmetry' ]]

s1.plot.hist(alpha=0.75)
s2.plot.hist(alpha=0.75)

plt.show()

