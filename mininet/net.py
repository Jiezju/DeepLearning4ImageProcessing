import matplotlib.pyplot as plt

from miniflow import *
from sklearn.utils import resample
from sklearn import datasets
import numpy as np

np.random.seed(42)

data = datasets.load_iris()
X_ = data.data
y_ = data.target
y_[y_ == 2] = 1  # 0 for virginica, 1 for not virginica
print(X_.shape, y_.shape)

# 可以理解为两层逻辑斯蒂回归串联，输入层4个特征，输出层2个预测（不是 virginica 的可能性），中间层这里设置为3。
n_features = X_.shape[1]
n_class = 1
n_hidden = 3

X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
t1 = Sigmoid(l2)
cost = MSE(y, t1)

# 随机初始化参数值
W1_0 = np.random.random(X_.shape[1] * n_hidden).reshape([X_.shape[1], n_hidden])
W2_0 = np.random.random(n_hidden * n_class).reshape([n_hidden, n_class])
b1_0 = np.random.random(n_hidden)
b2_0 = np.random.random(n_class)

# 将输入值带入算子
feed_dict = {
    X: X_, y: y_,
    W1: W1_0, b1: b1_0,
    W2: W2_0, b2: b2_0
}

# 训练参数
# 这里训练100轮（eprochs），每轮抽4个样本（batch_size）训练150/4次（steps_per_eproch）,学习率 0.1
epochs = 100
m = X_.shape[0]
batch_size = 4
steps_per_epoch = m // batch_size
lr = 0.1

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

l_Mat_W1 = [W1_0]
l_Mat_W2 = [W2_0]

l_loss = []
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)
        X.value = X_batch
        y.value = y_batch

        forward_and_backward(graph)
        sgd_update(trainables, lr)
        loss += graph[-1].value

    l_loss.append(loss)
    if i % 10 == 9:
        print("Eproch %d, Loss = %1.5f" % (i, loss))

plt.plot(l_loss)
plt.title("Cross Entropy value")
plt.xlabel("Eproch")
plt.xlabel("Loss")

# 用模型预测所有数据
X.value = X_
y.value = y_
for n in graph:
    n.forward()
plt.plot(graph[-2].value.ravel())
plt.title("Predict for all 150 Iris data")
plt.xlabel("Sample ID")
plt.ylabel("Probability for not a virginica")
plt.show()
