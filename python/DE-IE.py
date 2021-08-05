import cv2
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter, uniform_filter

# 窗口大小
kernel_size = 3
popu_size = 30
max_iter = 20
dim = 4
F = 0.5
CR = 0.5


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


# 可以实现灰度变换
def ok_transform(img, a, b, c, k, k_size):
    new_img = copy.deepcopy(img).astype(np.float)
    D = np.mean(new_img)
    M = cv2.blur(new_img, (k_size, k_size))
    S = window_stdev(new_img, k_size)
    # S = generic_filter(new_img, np.std, size=k_size)
    new_img = k * D / (S + b) * (new_img - c * M) + np.power(M, a)
    return np.uint8(np.clip(new_img, 0, 255))  # 转换成0-255之间的数字


# 计算图像的熵
def calc_entropy(img):
    rows = img.shape[0]
    cols = img.shape[1]
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
    rows, cols = img.shape
    E = np.sum(img)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    absx = cv2.convertScaleAbs(sobelX)
    absy = cv2.convertScaleAbs(sobelY)
    sobel = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    ret, threshold = cv2.threshold(
        sobel, 100, 255, cv2.THRESH_BINARY_INV
    )  # 转化为二值化图像，图像阈值为100
    n_edgels = len(threshold[threshold == 255])  # 二值化图像大于阈值的像素个数
    entropy = calc_entropy(img)
    return math.log(math.log(E)) * n_edgels / (rows * cols) * entropy


def DEIE(popu_size, max_iter, dim, img, calc_fitness, k_size, F, CR):
    X = np.array(generate_population(popu_size))  # 初始化粒子群位置
    V = [[0 for i in range(dim)] for j in range(popu_size)]
    U = [[0 for i in range(dim)] for j in range(popu_size)]

    print("====================初始化适应度====================")
    fitness = [
        calc_fitness(ok_transform(img, x[0], x[1], x[2], x[3], k_size)) for x in X
    ]
    print("====================适应度初始化完成====================")
    print("fitness", fitness)
    best = X[np.argmax(fitness)]  # 初始化全局最优位置
    best_history = [best.tolist()]
    figure_count = 0
    print("====================开始迭代====================")
    for iter in range(max_iter):
        print("====================第{}次迭代====================".format(iter))
        for i in range(popu_size):
            img_out = ok_transform(img, X[i][0], X[i][1], X[i][2], X[i][3], k_size)
            figure_count = figure_count + 1
            cv2.imwrite("result/rice/figure{}.png".format(figure_count), img_out)
            new_fitness = calc_fitness(img_out)
            if new_fitness > fitness[i]:
                best = X[i].copy()

            # mutation
            # r1 r2 r3 random and different
            r1 = random.randrange(popu_size)
            while r1 == i:
                r1 = random.randrange(popu_size)
            r2 = random.randrange(popu_size)
            while r2 == i or r2 == r1:
                r2 = random.randrange(popu_size)
            r3 = random.randrange(popu_size)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = random.randrange(popu_size)

            for j in range(dim):
                V[i][j] = X[r1][j] + F * (X[r2][j] - X[r3][j])
                # limit of bound
                # V[i][j] = (
                #     x_low_bound if V[i][j] < x_low_bound else V[i][j]
                # )
                # V[i][j] = (
                #     x_up_bound if V[i][j] > x_up_bound else V[i][j]
                # )

            # crossover
            for j in range(dim):
                U[i][j] = V[i][j] if random.random() <= CR or j == random.randrange(dim) else X[i][j]
                # limit of bound
                # U[i][j] = (x_low_bound if U[i][j] < x_low_bound else U[i][j]
                # )
                # U[i][j] = (
                #     x_up_bound if U[i][j] > x_up_bound else U[i][j]
                # )

            # select
            # x_fitness = calc_fitness(X[i])
            u_fitness = calc_fitness(ok_transform(img, U[i][0], U[i][1], U[i][2], U[i][3], k_size))
            if u_fitness > fitness[i]:
                X[i] = U[i].copy()
                # for j in range(dim):
                #     X[i][j] = U[i][j]

            # x_fitness = calc_fitness(X[i])
            # if x_fitness > global_fitness:
            #     global_fitness = x_fitness
            #     global_solution = X[i]
        print("best: ", best)
        best_history.append(best.tolist())
        print("更新后粒子的位置：")
        print(X)
        # X[:, 0] = np.clip(X[:, 0], 0, 1.5)
        # X[:, 1] = np.clip(X[:, 1], 0, 0.5)
        # X[:, 2] = np.clip(X[:, 2], 0, 1)
        # X[:, 3] = np.clip(X[:, 3], 0.5, 1.5)
        # print("修改范围后粒子的位置：")
        # print(X)
    print("history of best:")
    for best in best_history:
        print(best)
    return best


# 读入原始图像
img = cv2.imread("../images/rice.png", 0)
DEIE(popu_size, max_iter, dim, img, calc_fitness, kernel_size, F, CR)

# 通过窗口展示图片 第一个参数为窗口名 第二个为读取的图片变量
# cv2.imshow("img", img)


# cv2.waitKey(0)
# cv2.destroyAllWindows()