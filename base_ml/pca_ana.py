from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

'''
PCA:
    数据降维
    最大化投影数据方差；最小化投影损失
'''

pca = PCA()

# import data
data = datasets.load_iris()
df = pd.DataFrame(data.data)
df.columns = data.feature_names
df['species'] = [data['target_names'][x] for x in data.target]
df_sub = df[data.feature_names[:3]]
pca.fit(df_sub)

pca_result = pca.transform(df_sub)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
ax.scatter(pca_result[:, 0], pca_result[:, 1], c=data.target, cmap=plt.cm.Set3)

# 显示投影平面
plane_show_size_ratio = 5
plane_show_shift = df_sub.mean().values
pca_score = pca.explained_variance_ratio_
v = pca.components_
l_pca_axis = v.T * plane_show_size_ratio
l_pca_plane = []

for pca_axis in l_pca_axis:
    l_pca_plane.append(np.r_[pca_axis[:2], -pca_axis[1::-1]].reshape(2, 2))

fig = plt.figure(figsize=(4, 4))
ax = Axes3D(fig, rect=(0., 0., 0.95, 1.), elev=150, azim=-34)
ax.scatter(df_sub.values[:, 0], df_sub.values[:, 1], df_sub.values[:, 2], '.', c=data.target, cmap=plt.cm.Set3)
ax.plot_surface(l_pca_plane[0] + plane_show_shift[0],
                l_pca_plane[1] + plane_show_shift[1],
                l_pca_plane[2] + plane_show_shift[2], alpha=0.1)

plt.show()
