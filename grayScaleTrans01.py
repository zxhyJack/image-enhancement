import cv2
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter, uniform_filter

# 灰度化处理
# img1=cv2.imread('girl.png',0)

# 灰度化处理：此灰度化处理用于图像二值化
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 窗口大小
kernel_size = 3
popu_size = 10
max_iter = 10
dim = 4


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


# 窗口标准差
def window_stdev(X, window_size):
    (
        r,
        c,
    ) = X.shape
    X += np.random.rand(r, c) * 1e-6
    c1 = uniform_filter(X, window_size, mode="reflect")
    c2 = uniform_filter(X * X, window_size, mode="reflect")
    return np.sqrt(c2 - c1 * c1)


# 灰度变换
def transform(img, a, b, c, k, n):
    newImg = copy.deepcopy(img)
    rows, cols = img.shape
    D = 0
    for i in range(rows):
        for j in range(cols):
            D += img[i][j][0]
    D = D / (rows * cols)
    for i in range(rows):
        for j in range(cols):
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
    rows, cols = img.shape
    C = 3
    pad = kernel_size // 2
    new_img = np.zeros((rows + pad * 2, cols + pad * 2, C), dtype=np.float)
    new_img[pad : pad + rows, pad : pad + cols] = img.copy().astype(np.float)
    # tmp = new_img.copy()
    M = cv2.blur(img, (3, 3))
    D = np.sum(new_img) / 3
    D = D / (rows * cols)
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


# 可以实现变换
def ok_transform(img, a, b, c, k, k_size):
    new_img = copy.deepcopy(img).astype(np.double)
    D = np.mean(new_img)
    M = cv2.blur(new_img, (3, 3))
    S = window_stdev(new_img, k_size)
    # S = generic_filter(new_img, np.std, size=k_size)
    new_img = k * D / (S + b) * (new_img - c * M) + np.power(M, a)
    return np.uint8(np.clip(new_img, 0, 255))  # 转换成0-255之间的数字


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
            val = img[i][j]
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
    E = np.sum(img)
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
    return math.log(math.log(E)) * n_edgels / (rows * cols) * entropy


def PSOIE(popu_size, max_iter, dim, img, calc_fitness, k_size, w=0.6, c1=2, c2=2):
    X = np.array(generate_population(popu_size))  # 初始化粒子群位置
    V = np.random.rand(popu_size, dim)  # 初始化粒子群速度
    pbest = X  # 初始化个体最优位置
    print("===============初始化适应度===============")
    fitness = np.array(
        [calc_fitness(ok_transform(img, x[0], x[1], x[2], x[3], k_size)) for x in pbest]
    )
    print("===============适应度初始化完成===============")
    print(fitness)
    gbest = X[np.argmax(fitness)]  # 初始化全局最优位置
    print("gbest:", gbest)
    print("================开始迭代===============")
    figure_count = 0
    for iter in range(max_iter):
        print("第========{}========次迭代".format(iter))
        for i in range(popu_size):
            print("第{}个粒子".format(i))
            out = ok_transform(img, X[i][0], X[i][1], X[i][2], X[i][3], k_size)
            figure_count = figure_count + 1
            # cv2.imshow('Figure.{}'.format(figure_count), out)
            cv2.imwrite("result/cameraman/figure{}.png".format(figure_count), out)
            new_fitness = calc_fitness(out)
            if new_fitness > fitness[i]:
                pbest[i] = X[i]
                fitness[i] = new_fitness
            if new_fitness > np.max(fitness):
                gbest = X[i]
            print("pbest: ", pbest[i])
        print("gbest: ", gbest)
        r1 = np.random.rand(popu_size, dim)
        r2 = np.random.rand(popu_size, dim)
        # 更新速度和权重
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        print("更新后粒子的位置：")
        print(X)
        # print("更新后粒子的速度：")
        # print(V)
    return gbest


# 读入原始图像
img = cv2.imread("./imageData/cameraman.tif", 0)
PSOIE(popu_size, max_iter, dim, img, calc_fitness, kernel_size)

# 通过窗口展示图片 第一个参数为窗口名 第二个为读取的图片变量
# cv2.imshow("img", img)


# cv2.waitKey(0)
# cv2.destroyAllWindows()