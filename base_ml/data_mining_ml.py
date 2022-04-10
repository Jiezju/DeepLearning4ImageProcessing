import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# import data
data = datasets.load_iris()

# pandas 数据处理
df = pd.DataFrame(data.data)

df.columns = data.feature_names
df['species'] = [data['target_names'][x] for x in data.target]

'''
数据划分，重点时保持 训练集和测试集
'''

# data split 80% train 20% test
df_train, df_val = train_test_split(df, train_size=0.8, random_state=0)

# 提取特征
X_train = df_train.drop(['species'], axis=1)
X_val = df_val.drop(['species'], axis=1)

# 提取 分类结果
Y_train = df_train['species']
Y_val = df_train['species']

# 设定分布 X_scaler，用训练集估计(fit)分布，然后对验证集进行转换(transform)
X_scaler = StandardScaler()
X_trainT = X_scaler.fit_transform(X_train)
X_valT = X_scaler.transform(X_val)

# 这里将保证训练集是标准正态分布，验证集不一定满足这个条件，但不会差很多
print(X_trainT.mean(axis=0), X_trainT.var(axis=0))
print(X_valT.mean(axis=0), X_valT.var(axis=0))

'''
manual realize
'''
C = 1  # 正则化系数
alpha = 0.1  # 学习率
Y_lab = Y_val.values
y_train = np.zeros(Y_lab.shape).astype(np.double)
y_train[Y_lab != 'setosa'] = 0.0
y_train[Y_lab == 'setosa'] = 1.0
X_1 = np.hstack([X_train.values, np.ones_like(y_train).reshape(len(y_train), 1).astype(np.double)])

# 第一步 omega 随即初始化
np.random.seed(42)
omega = np.random.random(X_1.shape[1]).reshape(5, 1)

for i in range(10):
    # 第二步 前向传播
    y_hat = 1 / (1 + np.exp(-X_1.dot(omega)) + 1e-5)

    # 第三步 计算误差
    dl = X_1.T.dot(C * (y_train.reshape(-1, 1) - y_hat)) + omega

    # update
    omega += np.double(alpha) * dl

print(omega)

# 交叉验证 sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
parameters = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
model = LogisticRegression()
clf = GridSearchCV(model, parameters, cv=10)
clf.fit(X_1, y_train)
# print(clf.cv_results_)
# plt.plot( np.log10(np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])), clf.cv_results_['mean_train_score'], label="train")
plt.plot( np.log10(np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])), clf.cv_results_['mean_test_score'], label="test")
plt.xlabel("log10 C")
plt.legend()
plt.ylabel("Cross Validation Accuracy")
plt.show()
