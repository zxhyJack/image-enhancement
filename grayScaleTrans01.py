import cv2
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# 读入原始图像
img = cv2.imread("./imageData/cameraman.tif", 1)

# 灰度化处理
# img1=cv2.imread('girl.png',0)

# 灰度化处理：此灰度化处理用于图像二值化
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 窗口大小
kernel_size = 3


def generate_population(popu_size):
    x = []
    for i in range(popu_size):
        x.append(
            [
                random.uniform(0, 1.5),
                random.uniform(0, 0.5),
                random.random(),
                random.uniform(0.5, 1.5),
            ]
        )
    return x


# 灰度变换
def transform(img, a, b, c, k, n):
    newImg = copy.deepcopy(img)
    rows = img.shape[0]
    cols = img.shape[1]
    D = 0
    for i in range(rows):
        for j in range(cols):
            D += img[i][j][0]
    D = D / (rows * cols)
    for i in range(rows):
        for j in range(cols):
            # gamma[i][j] = 3 * pow(gamma[i][j], 0.8)
            m = 0
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    if x < 0 or y < 0 or x > 255 or y > 255:
                        continue
                    m += newImg[x][y]
            m = m / (n * n)
            sigma = 0
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    if x < 0 or y < 0 or x > 255 or y > 255:
                        continue
                    sigma += np.square(newImg[x][y] - m)
            sigma = np.sqrt(sigma / (n * n))
            newImg[i][j] = k * D * (newImg[i][j] - c * m) / (sigma + b) + np.power(m, a)
    return newImg


def customize_transform(img, a, b, c, k, kernel_size):
    # new_img = copy.deepcopy(img).astype(np.float)
    rows = img.shape[0]
    cols = img.shape[1]
    C = 3
    pad = kernel_size // 2
    new_img = np.zeros((rows + pad * 2, cols + pad * 2, C), dtype=np.float)
    new_img[pad : pad + rows, pad : pad + cols] = img.copy().astype(np.float)
    # tmp = new_img.copy()
    M = cv2.blur(img, (3, 3))
    D = np.sum(new_img) / 3
    D = D / (rows * cols)
    print(D)
    for i in range(rows):
        for j in range(cols):
            sigma = np.sqrt(
                np.mean(
                    np.square(
                        new_img[i : i + kernel_size, j : j + kernel_size] - M[i][j]
                    )
                )
            )
            new_img[i][j] = k * D * (new_img[i][j] - c * M[i][j]) / (
                sigma + b
            ) + np.power(M[i][j], a)
    return new_img[pad : pad + rows, pad : pad + cols].astype(np.uint8)


# 计算图像的熵
def calc_entropy(img):
    rows = img.shape[0]
    cols = img.shape[1]
    # tmp = []
    # for i in range(256):
    #     tmp.append(0)
    tmp = [0 for i in range(256)]
    val = 0
    entropy = 0
    # 统计不同灰度值像素的个数
    for i in range(rows):
        for j in range(cols):
            val = img[i][j][0]
            tmp[val] += 1
    # 计算每个灰度值像素出现的概率
    P = [0 for i in range(256)]
    P = [float(i / (rows * cols)) for i in tmp]
    for i in range(256):
        if P[i] == 0:
            continue
        else:
            entropy = float(entropy - P[i] * math.log2(P[i]) / 1)
    return entropy


# 计算适应度
def calc_fitness(img):
    rows = img.shape[0]
    cols = img.shape[1]
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    absx = cv2.convertScaleAbs(sobelX)
    absy = cv2.convertScaleAbs(sobelY)
    sobel = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    ret, threshold = cv2.threshold(
        img, 100, 255, cv2.THRESH_BINARY_INV
    )  # 转化为二值化图像，图像阈值为100
    n_edgels = len(threshold[threshold == 255])  # 二值化图像大于阈值的像素个数
    entropy = calc_entropy(img)
    return math.log(math.log(rows * cols)) * n_edgels / (rows * cols) * entropy


def PSOIE(popu_size, max_iter, dim, img, calc_fitness, w=0.6, c1=2, c2=2):
    X = np.array(generate_population(popu_size))  # 初始化粒子群位置
    V = np.random.rand(popu_size, dim)  # 初始化粒子群速度
    pbest = X  # 初始化个体最优位置
    fitness = np.array(
        [
            calc_fitness(customize_transform(img, x[0], x[1], x[2], x[3], kernel_size))
            for x in pbest
        ]
    )
    gbest = X[np.argmax(fitness)]  # 初始化全局最优位置
    for iter in range(max_iter):
        for i in range(popu_size):
            new_fitness = calc_fitness(
                customize_transform(
                    img, X[i][0], X[i][1], X[i][2], X[i][3], kernel_size
                )
            )
            if new_fitness > fitness[i]:
                pbest[i] = X[i]
                fitness[i] = new_fitness
            if new_fitness > np.max(fitness):
                gbest = X[i]
            print(new_fitness, fitness[i])
        r1 = np.random.rand(popu_size, dim)
        r2 = np.random.rand(popu_size, dim)
        # 更新速度和权重
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        print(X)
    return gbest


bestparam = PSOIE(10, 10, 4, img, calc_fitness)
print(bestparam)

# 通过窗口展示图片 第一个参数为窗口名 第二个为读取的图片变量
cv2.imshow("img", img)
cv2.imshow(
    "newImg",
    customize_transform(
        img, bestparam[0], bestparam[1], bestparam[2], bestparam[3], kernel_size
    ),
)
# cv2.imshow("sobelX", sobelX)
# cv2.imshow("sobelY", sobelY)
# cv2.imshow("sobel", sobel)
# cv2.imshow("threshold", threshold)


# 暂停cv2模块 不然图片窗口一瞬间即就会消失 观察不到
cv2.waitKey(0)