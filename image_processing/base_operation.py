import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('messi5.jpg')

# show
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img[:, :, ::-1])
ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

img_small = cv2.resize(img, (360, 240))
cv2.imwrite('./out.png', img_small)

cv2.putText(img, "Messi", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
plt.imshow(img[:, :, np.array([2, 1, 0])])
plt.show()
