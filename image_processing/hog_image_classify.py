import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import scipy.stats as stats
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix, auc, roc_auc_score
import matplotlib.pyplot as plt
import sklearn
from skimage.feature import hog
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
import time
from scipy.ndimage.measurements import label
import numpy as np
import functools
import pickle

'''
将 txt 文件转为 csv 文件格式
'''
#  ls ./data/*vehicles/*/* >> data.txt
with open("../data.txt", "r") as f:  # 打开文件
    l_samp = f.read().splitlines()  # 读取文件

M_ClassDict = {"non-vehicles": 0, "vehicles": 1}
pd_SampClass = pd.DataFrame({
    "Sample": l_samp,
    "Class": list(map(lambda x: M_ClassDict[x], list(map(lambda x: x.split("/")[2], l_samp))))
})[['Sample', 'Class']]

# 划分训练集和测试集合
pd_SampClass_train, pd_SampClass_test = train_test_split(pd_SampClass, test_size=0.33, random_state=42)
print(pd_SampClass_train.head())

fig = plt.figure(figsize=(12, 6))
for i in range(5):
    image = cv2.imread(pd_SampClass_train['Sample'].iloc[i])
    image = image[:, :, ::-1]
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(image)
    ax.set_title(pd_SampClass_train['Class'].iloc[i])

# test hog
fig = plt.figure(figsize=(20, 10))

img = cv2.imread("../data/vehicles/GTI_Far/image0000.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for i1, pix_per_cell in enumerate([6, 8, 10]):
    for i2, cell_per_block in enumerate([2, 3]):
        for i3, orient in enumerate([6, 8, 9]):
            # pixels_per_cell 使用多少像素作为一个网格，值越高，切出的网格越少，检测结果越粗糙
            features, hog_image = hog(img_gray, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      orientations=orient, visualize=True, feature_vector=False
                                      )
            # print(features.shape)
            ax = fig.add_subplot(3, 6, i1 * 6 + i2 * 3 + i3 + 1)
            ax.imshow(hog_image, 'gray')
            ax.set_title("Pix%d_C%d_Ori%d" % (pix_per_cell, cell_per_block, orient))

'''
1. 将图片的特征尽可能简化，比如比较简单的特征：轮廓，几何特征等等
2. 将这些特征进行提取，通过机器学习分类器进行
'''
# 这里只看灰度图的轮廓，不考虑颜色。如果需要考虑，这里可以继续添加
l_colorSpace = [cv2.COLOR_BGR2GRAY]
l_names = ["GRAY"]
l_len = [1]


def get_hog_features(img, orient, pix_per_cell=8, cell_per_block=2,
                     vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image

    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def get_features(img, pix_per_cell=8, cell_per_block=2, orient=9, getImage=False, inputFile=True, feature_vec=True):
    # print(img)
    l_imgLayers = []
    for cs in l_colorSpace:
        if inputFile:
            l_imgLayers.append(cv2.cvtColor(cv2.imread(img), cs))
        else:
            l_imgLayers.append(cv2.cvtColor(img, cs))

    l_hog_features = []
    l_images = []
    for feature_image in l_imgLayers:
        hog_features = []
        n_channel = 1
        if len(feature_image.shape) > 2:
            n_channel = feature_image.shape[2]
        for channel in range(n_channel):
            featureImg = feature_image
            if n_channel > 2:
                featureImg = feature_image[:, :, channel]

            vout, img = get_hog_features(featureImg,
                                         orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=feature_vec)
            if getImage:
                l_images.append(img)
            # print(featureImg.shape, vout.shape)
            hog_features.append(vout)

        l_hog_features.append(list(hog_features))

    if getImage:
        return l_images
    else:
        return functools.reduce(lambda x, y: x + y, l_hog_features)


# 对划分好的训练集/测试集提取图像信息，计算 hog 值，存储中间结果
if os.path.isfile("./X_train.npy") == 0:
    l_X_train = []
    l_X_test = []
    for r in tqdm(pd_SampClass_train.iterrows(), total=10, ncols=100):
        l_X_train.append(np.array(get_features(r[1]['Sample'])).ravel())

    for r in tqdm(pd_SampClass_test.iterrows()):
        l_X_test.append(np.array(get_features(r[1]['Sample'])).ravel())

    X_train = np.array(l_X_train)
    X_test = np.array(l_X_test)
    np.save("./X_train.npy", X_train)
    np.save("./X_test.npy", X_test)
else:
    X_train = np.load("./X_train.npy")
    X_test = np.load("./X_test.npy")

y_train = pd_SampClass_train['Class'].values
y_test = pd_SampClass_test['Class'].values

# 开始模型训练
X_scalerM = StandardScaler()
X_trainT = X_scalerM.fit_transform(X_train)
X_testT = X_scalerM.transform(X_test)

X_trainTs, y_trainTs = sklearn.utils.shuffle(X_trainT, y_train)

svc = SVC(random_state=0, C=1)
t = time.time()
svc.fit(X_trainTs, y_trainTs)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')

# 查看测试集准确率
print('Test Accuracy of SVC = ', round(svc.score(X_testT, y_test), 4))
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_testT[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()

# 查看 auc 值
pred = svc.predict(X_testT)
print("AUC for Merge dataset = %1.2f,\n" % (roc_auc_score(pred, y_test)))
print(confusion_matrix(pred, y_test))

plt.show()
