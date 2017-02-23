import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


wheat = pd.read_csv("C:/Users/mckinns/Documents/GitHub/DAT210x/Module3/Datasets/wheat.data", 
                              index_col=0, header=0)

fig = plt.figure()
# area,perimeter and asymmetry features. Be sure to use the
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')

ax.scatter(wheat.area, wheat.perimeter, wheat.asymmetry, c='r', marker='.')
plt.show()

fig = plt.figure()
#
#  width, groove and length features. Be sure to use the
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('width')
ax.set_ylabel('groove')
ax.set_zlabel('length')

ax.scatter(wheat.width, wheat.groove, wheat.length, c='green', marker='.')
plt.show()


