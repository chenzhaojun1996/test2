import cv2
import  matplotlib.pyplot  as plt
import numpy as np


img = cv2.imread("dige.png")
cv2.imshow("img", img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 定义卷积核大小
# 使用erode进行腐蚀，iteration定义迭代次数
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow("erosion", erosion)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 拿个白圆黑背景的图，来增加腐蚀的迭代次数，观察圆越来越小
pie = cv2.imread("pie.png")
cv2.imshow("pie",pie)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 膨胀与腐蚀相反，把物体边缘扩大，图片是迪哥文字，使用后，会使文字变粗
kernel = np.ones((5,5), np.uint8)
erosion1 = cv2.erode(pie, kernel, iterations=1)
erosion2 = cv2.erode(pie, kernel, iterations=2)
erosion3 = cv2.erode(pie, kernel, iterations=10)
res = np.hstack((erosion1, erosion2, erosion3))
cv2.imshow("res", res)
cv2.waitKey(1000)
cv2.destroyAllWindows()

#  先腐蚀后膨胀
img = cv2.imread("dige.png")
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(erosion, kernel,iterations=1)
cv2.imshow("dige_dilate", dilate)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# 开运算，先腐蚀后膨胀

img = cv2.imread("dige.png")
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN,kernel)
cv2.imshow("openging", opening)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# 闭运算，先膨胀后腐蚀
img = cv2.imread("dige.png")
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel)
cv2.imshow("closing", closing)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# 梯度运算，其实就相当于：梯度=膨胀-腐蚀
# 通过梯度运算得到 物体的边缘
pie = cv2.imread("pie.png")
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(pie, kernel,iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)
res = np.hstack((dilate, erosion))
cv2.imshow("res", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 梯度运算具体实现
gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("gradient", gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 礼帽和黑帽
# 礼帽= 原始输入 - 开运算cv2.morphologyEx的参数改为 cv2.MORPH_PATHAT
#  具体运用见API调用
# 黑帽运算 = 闭运算 - 原始输入



