import cv2
import copy
import time
import numpy as np
from scipy.ndimage.filters import generic_filter


def transform(img, a, b, c, k, k_size):
    new_img = copy.deepcopy(img).astype(np.double)
    rows, cols, C = img.shape
    D = np.mean(new_img)
    M = cv2.blur(new_img, (3, 3))
    S = generic_filter(new_img, np.std, size=3)
    print(D)
    print(M[:5, :5])
    print(S[:5, :5])
    pad = k_size // 2
    # new_img = np.zeros((rows + pad * 2, cols + pad * 2, C), dtype=np.double)
    # new_img[pad : pad + rows, pad : pad + cols] = img.copy().astype(np.double)
    # tmp = new_img.copy()
    # cv2.imshow('M', M)
    # sigma = new_img.copy()
    # M = new_img.copy()
    # M = cv2.blur(img, (3, 3))
    # D = np.sum(new_img) / 3
    # D = D / (rows * cols)
    # for i in range(rows):
    #     for j in range(cols):
    #         M[pad + i][pad + j] = np.mean(tmp[i : i + k_size, j : j + k_size])
    # for i in range(rows):
    #     for j in range(cols):
    #         # M[pad + i][pad + j] = np.mean(tmp[i : i + k_size, j : j + k_size])
    #         sigma[pad + i][pad + j] = np.sqrt(
    #             np.mean(
    #                 np.square(tmp[i : i + k_size, j : j + k_size] - M[pad + i][pad + j])
    #             )
    #         )
    # for i in range(rows):
    #     for j in range(cols):
    #         # for l in range(C):
    #         # m = np.mean(tmp[i : i + k_size, j : j + k_size])
    #         sigma = np.sqrt(
    #             np.mean(np.square(tmp[i : i + k_size, j : j + k_size] - M[i][j]))
    #         )
    #         new_img[pad + i][pad + j] = k * D * (tmp[pad + i][pad + j] - c * M[i][j]) / (
    #             sigma + b
    #         ) + np.power(M[i][j], a)
    new_img = k * D / (S + b) * (new_img - c * M) + np.power(M, a)

    # print(M.astype(np.uint8)[0:5][0:5])
    # print(cv2.blur(img, (3, 3))[0:4][0:4])
    # cv2.imshow('M',M.astype(np.uint8))
    # cv2.imshow('sigma',sigma.astype(np.uint8))
    # cv2.imshow('new img',new_img.astype(np.uint8))
    # return new_img[pad : pad + rows, pad : pad + cols].astype(np.uint8)
    return np.uint8(np.clip(new_img,0,255))

    # for i in range(rows):
    #     for j in range(cols):
    #       new_img[i][j] = 3 * pow(new_img[i][j], 0.8)
    #       m = 0
    #       count=0
    #       for x in range(i - 1, i + 2):
    #           for y in range(j - 1, j + 2):
    #               if x < 0 or y < 0 or x > 255 or y > 255:
    #                   continue
    #               count = count + 1
    #               m += new_img[x][y]
    #       m = m / count
    #       new_img[i][j] = m / count
    #       sigma = 0
    #       count = 0
    #       for x in range(i - 1, i + 2):
    #           for y in range(j - 1, j + 2):
    #               if x < 0 or y < 0 or x > 255 or y > 255:
    #                   continue
    #               sigma += np.square(new_img[x][y] - M[i][j])
    #               count = count + 1
    #       sigma = np.sqrt(sigma / count)
    #       new_img[i][j] = k * D * (new_img[i][j] - c * M[i][j]) / (sigma + b) + np.power(M[i][j], a)

    #       return new_img.astype(np.uint8)


img = cv2.imread("./imageData/rice.png", 1)
start = time.time()
new_img = transform(img, 0.91, 0.3, 0.86, 1.3, 3)
end = time.time()
print(end - start)
print("new img", new_img[:5, :5])

cv2.imshow("original", img)
cv2.imshow("new_img", new_img)
cv2.waitKey(0)
