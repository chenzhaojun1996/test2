import cv2
import  matplotlib.pyplot  as plt
import numpy as np

# 展示图像函数
def show_img(img_name, img_src):
    cv2.imshow(img_name, img_src)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


img = cv2.imread("lena.jpg")
show_img("lena", img)

# 白到黑是正数，因为梯度计算是从右边到计算，所以要注意负值问题，负值被截断为0
#  1,0 是x方向, kszie=3表示卷积核大小为3，sobel算子其实就是卷积核矩阵
# cv2.CV_64F表示的是深度，通常是-1，好吧，这边的确不太懂
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
show_img("sobelx", sobelx)

# 将结果取绝对值
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
print(sobelx)
show_img("sobelx", sobelx)

# y方向的 从下到上计算
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobelx)
show_img("sobely", sobely)

# x，y方向加起来 （Gx，Gx权重，Gy，Gy权重，偏置项一般为0）
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely,0.5, 0)
show_img("sobel_xy", sobelxy)

# 不建议直接计算x，y方向的：将参数设置为1,1，这样效果不好，结果重叠

# 不同算子的差异
img = cv2.imread("lena.jpg")
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
show_img("sobelxy", sobelxy)


scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
show_img("scharryx", scharrxy)


laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
show_img("laplacian", laplacian)
