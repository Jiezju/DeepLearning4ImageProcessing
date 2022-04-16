import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

img = cv2.imread('./messi5.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ball = img[280:340, 330:390]
plt.imshow(ball[:, :, ::-1])

# 1. RGB 像素分布
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(332)

# 组合 计算 各个 pair 的颜色组合
for r in ball.reshape(-1, 3):
    ax.plot(r[1], r[0], '.', c=(r[0] / 255., r[1] / 255., r[2] / 255.))

ax = fig.add_subplot(333)
for r in ball.reshape(-1, 3):
    ax.plot(r[2], r[0], '.', c=(r[0] / 255., r[1] / 255., r[2] / 255.))

ax = fig.add_subplot(336)
for r in ball.reshape(-1, 3):
    ax.plot(r[2], r[1], '.', c=(r[0] / 255., r[1] / 255., r[2] / 255.))

for i, color in enumerate(['Red', "Green", "Blue"]):
    ax = fig.add_subplot(3, 3, i * 3 + i + 1)
    ax.text(5, 5, color)
    ax.plot(0, 0)
    ax.plot(10, 10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# 黄色和蓝黑色均为足球，设置组合规则
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow((ball[:, :, 0] > 200))
temp = (ball[:, :, 0] > 130) + (ball[:, :, 0] < 50)
ax2.imshow((ball[:, :, 0] > 130) + (ball[:, :, 0] < 50))
ax3.imshow((ball[:, :, 0] > 130) + (ball[:, :, 0] < 50) + (ball[:, :, 1] < 120))

# 分割足球
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

img1 = ((img[:, :, 0] > 130) + (img[:, :, 0] < 50) + (img[:, :, 1] < 120)).astype(np.uint8)
ax1.imshow(img)
ax2.imshow(img1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img_e = cv2.erode(img1, kernel, iterations=2)
img_de = cv2.dilate(img_e, kernel, iterations=8)
img_ede = cv2.erode(img_de, kernel, iterations=3)
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(img_e)
ax2.imshow(img_de)
ax3.imshow(img_ede)
ax1.set_title(u"first erode 2 pixels")
ax2.set_title(u"then dilate 8 pixels")
ax3.set_title(u"finally erode 3 pixels")

# 使用梯度获取封闭边缘，并强化边缘
sobelx = cv2.Sobel(img_ede * 255, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img_ede * 255, cv2.CV_64F, 0, 1)
img_sob = np.sqrt(sobelx ** 2 + sobely ** 2).astype(np.uint8)
plt.imshow(img_sob)

gray = img_sob
# 针对边缘进行检测和去噪
canny = cv2.Canny(gray, 200, 300)
gray = cv2.medianBlur(gray, 5)
np_hc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                         param1=200,
                         param2=10,
                         minRadius=20,
                         maxRadius=30)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img_sob, "gray")
img_tmp = np_hc
plt.show()
for i in range(np_hc.shape[1]):
    img_tmp = cv2.circle(img, (int(np_hc[0, i, 0]), int(np_hc[0, i, 1])), int(np_hc[0, i, 2]), (255, 0, 0), 8)

plt.imshow(img_tmp)

plt.show()
