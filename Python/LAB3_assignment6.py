import pandas as pd
import matplotlib.pyplot as plt

wheat = pd.read_csv("C:/Users/mckinns/Documents/GitHub/DAT210x/Module3/Datasets/wheat.data", 
                              index_col=0, header=0)

wheat.corr()

plt.imshow(wheat.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(wheat.columns))]
plt.xticks(tick_marks, wheat.columns, rotation='vertical')
plt.yticks(tick_marks, wheat.columns)

plt.show()


