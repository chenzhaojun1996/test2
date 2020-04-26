import cv2
import matplotlib.pyplot as plt
import numpy as np


# 使用cv.read展示图像函数
def show_img(img_name, img_src):
    cv2.imshow(img_name, img_src)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


# calcHist显示直方图，第二个参数为[0][1][2]对应通道BGR
img = cv2.imread("cat.jpg", 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("cat.jpg")
# 隐藏坐标轴
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.hist(img.ravel(), 256)
plt.title("hist")
plt.show()



# BGR三个通道显示对应的直方图
img = cv2.imread("cat.jpg")
color = ["b", "g", "r"]
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()


#  mask
mask = np.ones(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
show_img("mask", mask)
masked_img = cv2.bitwise_and(img, img, mask=mask)
show_img("masked_img", masked_img)
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
plt.plot(hist_full)
plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

# 均衡化
img = cv2.imread("clahe.jpg", 0)
plt.hist(img.ravel(), 256)
plt.show()
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(), 256)
plt.show()
res = np.hstack((img, equ))
show_img("res", res)

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(img)
res = np.hstack((img, equ, res_clahe))
show_img("res", res)


