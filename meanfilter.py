import cv2
import numpy as np

# mean filter
def mean_filter(img, K_size=3):
    H, W, C = img.shape
    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad : pad + H, pad : pad + W] = img.copy().astype(np.float)
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.mean(
                    tmp[y : y + K_size, x : x + K_size, c]
                )
    out = out[pad : pad + H, pad : pad + W].astype(np.uint8)
    return out


def customize_mean_filter(img):
    rows, cols, c = img.shape
    new_img = img.copy().astype(np.float)
    for i in range(rows):
        for j in range(cols):
            m = 0
            count = 0
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    if x == -1 or y == -1 or x == 256 or y == 256:
                        continue
                    count = count + 1
                    m = m + new_img[x][y]
            new_img[i][j] = m / count
            # new_img[i][j] = np.mean(new_img[i - 1 : i + 1, j - 1 : j + 1])
    return new_img.astype(np.uint8)


def customize_std_deviation_filter(img, k_size):
    rows, cols, c = img.shape
    new_img = img.copy().astype(np.float)
    C = 3
    pad = k_size // 2
    new_img = np.zeros((rows + pad * 2, cols + pad * 2, C), dtype=np.float)
    new_img[pad : pad + rows, pad : pad + cols] = img.copy().astype(np.float)
    tmp = new_img.copy()
    M = cv2.blur(img, (3, 3))
    for i in range(rows):
        for j in range(cols):
            # for c in range(C):
            #     new_img[i][j][c] = np.sqrt(
            #         np.mean(np.square(tmp[i : i + k_size, j : j + k_size, c] - M[i][j][c]))
            #     )
            sigma = np.sqrt(
                np.mean(np.square(tmp[i : i + k_size, j : j + k_size] - M[i][j]))
            )
            new_img[pad + i][pad + j] = sigma
    return new_img[pad : pad + rows, pad : pad + cols].astype(np.uint8)


# Read image
img = cv2.imread("./imageData/cameraman.tif")

# Mean Filter
# out = mean_filter(img, K_size=3)
out = customize_std_deviation_filter(img, 3)
print(out)

# Save result

# cv2.imwrite("out.jpg", out)
cv2.imshow("original", img)
cv2.imshow("out", out)

cv2.waitKey(0)

# cv2.destroyAllWindows()
