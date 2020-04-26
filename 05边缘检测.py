import cv2
import  matplotlib.pyplot  as plt
import numpy as np


# 展示图像函数
def show_img(img_name, img_src):
    cv2.imshow(img_name, img_src)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


car = cv2.imread("car.png", cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(car, 80, 250)
v2 = cv2.Canny(car, 50, 100)
res = np.hstack((v1, v2))
show_img("car_canny", res)
