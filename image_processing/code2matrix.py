import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from sklearn.cluster import KMeans

sns.set_style('white')

'''
基于提取区域的像素值的取值特性进行ROI区域提取
'''


img_QR = cv2.imread('./3.png')
img_QR = cv2.cvtColor(img_QR, cv2.COLOR_BGR2GRAY)

print(img_QR.shape)

# 看看取值情况，发现只有0（黑色）和 255（白色）
print(set(img_QR.ravel()))

# 预处理 二维码图像
img_jizhiQR = cv2.imread("./jizhi_qinding.png")
img_jizhiQR = img_jizhiQR[:, :, ::-1]

# 去掉边缘的白色
img_jizhiQR_gray = cv2.cvtColor(img_jizhiQR, cv2.COLOR_RGB2GRAY)
np_totalRow = np.arange(img_jizhiQR.shape[0])
# 全白部分 均值 为 255 ！！
idx_rowUsed = np_totalRow[img_jizhiQR_gray.mean(0) != 255]
idx_colUsed = np_totalRow[img_jizhiQR_gray.mean(1) != 255]
img_jizhiQR_rmBlank = img_jizhiQR[idx_rowUsed, :, :][:, idx_colUsed, :]

# 去掉蓝色部分 设置黑色图像，保留原来黑色区域为白色，则 提取有效了区域，去除了蓝色
img_jizhiQR_new = np.zeros([img_jizhiQR_rmBlank.shape[0], img_jizhiQR_rmBlank.shape[1]])
idx_black = (img_jizhiQR_rmBlank[:, :, 0] < 10) * (img_jizhiQR_rmBlank[:, :, 1] < 10) * (
            img_jizhiQR_rmBlank[:, :, 2] < 10)
img_jizhiQR_new[idx_black] = 255

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(img_jizhiQR)
ax2.imshow(img_jizhiQR_new)


# 直接采样
def QR_code_to_mat(img_input, num_out):
    np_rawToQr_idx = np.linspace(0, num_out - 0.1, img_input.shape[0]).astype(np.int)
    np_rawToQr_mesh = np.meshgrid(np_rawToQr_idx, np_rawToQr_idx)
    np_QR_counts = np.zeros([num_out, num_out])
    for row_idx in range(img_input.shape[0]):
        for col_idx in range(img_input.shape[1]):
            col_num_25 = np_rawToQr_mesh[0][row_idx][col_idx]
            row_num_25 = np_rawToQr_mesh[1][row_idx][col_idx]
            # print(row_idx, col_idx)
            if img_input[row_idx][col_idx] == 255:
                np_QR_counts[row_num_25][col_num_25] += 1

    return np_QR_counts


np_25_counts = QR_code_to_mat(img_QR, 25)
# sns.displot(np_25_counts.ravel())
plt.imshow(np_25_counts > 40, cmap='gray')
plt.show()


# 采用 pooling 方式 采样
def AvgPool(img_input, num_out):
    k_size = int(img_input.shape[0] / num_out) + 1
    op = nn.AvgPool2d(kernel_size=(k_size, k_size), stride=(k_size, k_size))
    x = t.tensor(img_input[np.newaxis, np.newaxis, :, :]).double()
    res = op(x).numpy()

    bigger_label = 1
    # 基于 kmeans 查找阈值
    kmeans = KMeans(n_clusters=2, random_state=0).fit(res.ravel().reshape(-1, 1))
    if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
        bigger_label = 0

    threshold = min(res.ravel()[kmeans.labels_ == bigger_label])
    return res[0, 0, :, :] > threshold


np_25_counts_pt = AvgPool(img_QR, 25)

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)

ax1.imshow(np_25_counts_pt, cmap="gray")
plt.show()
