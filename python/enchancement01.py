import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


img = cv.imread("./imageData/cameraman.tif")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
result = cv.equalizeHist(gray)
# grayHist = calcGrayHist(img)
# x = np.arange(256)

plt.subplot(221)
plt.imshow(gray, cmap=plt.cm.gray), plt.title("(a)"), plt.axis('off')
# plt.plot(x, grayHist, 'r', linewidth=2, c='black')
plt.subplot(222)
plt.imshow(result, cmap=plt.cm.gray), plt.title("(b)"), plt.axis('off')

plt.subplot(223)
plt.hist(img.ravel(), 256), plt.title("(c)")

plt.subplot(224)
plt.hist(result.ravel(), 256), plt.title("(d)")
# plt.xlabel("gray Label")
# plt.ylabel("number of pixels")
plt.show()
# cv.imshow("img", img)
# cv.waitKey()
