from sklearn.decomposition import PCA

print PCA

pca = PCA(n_components=2, svd_solver='full')

print pca

# numpy to generate random numbers
import numpy as np

# create a random data set
df = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
# check correlation
df.corr()

pca.fit(df)
PCA(copy=True, n_components=2, whiten=False)

T = pca.transform(df)

df.shape
(430, 6) # 430 Student survey responses, 6 questions..

T.shape
(430, 2) # 430 Student survey responses, 2 principal components..