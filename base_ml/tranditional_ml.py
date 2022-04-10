import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# import data
data = datasets.load_iris()

# print data
for k in data:
    print(k)
    print(data[k])  # list

# 特征数据的维度
print(data['data'].shape)
print(data['target'].shape)

# pandas 数据处理
df = pd.DataFrame(data.data)
print(df.head())
df.columns = data.feature_names
df['species'] = [ data['target_names'][x] for x in data.target]
print(df.head())

# 统计类别数量
df_cnt = df['species'].value_counts().reset_index() # 返回数据框
print(df_cnt)
sns.barplot(data=df_cnt, x='index', y='species')

# 极值，缺失值的判定
df.describe()

# 确认特征是否服从正态分布
# 如果存在线性关系则基本符合正态分布
# 特征 sepal length； sepal width 基本都来自于同一个正态分布
# 另两个特征明显是 两条直线组成，则可能来自于两个正态分布
for i in range(4):
    name = data.feature_names[i]
    ax = plt.subplot(2, 2, i+1)
    stats.probplot(df[name], plot=ax)
    ax.set_title(name)

# 基于 pivot table 计算各个特征的方差
df_pivot_table = pd.melt(df, id_vars=['species']).pivot_table(index=['species'], columns=['variable'], aggfunc=[np.mean, np.var])
print(df_pivot_table)

# 基于特定类别，特定特征进行正态分布
fig = plt.figure(figsize=(12, 4))

for i in range(3):
    name = data.target_names[i]
    ax = plt.subplot(1, 3, i+1)
    stats.probplot(df[df['species'] == name][data.feature_names[2]], plot=ax)
    ax.set_title(name)

plt.show()