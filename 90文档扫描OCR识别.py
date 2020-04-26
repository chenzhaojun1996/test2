import cv2
import matplotlib.pyplot  as plt
import numpy as np
import myutils


# 使用cv.read展示图像函数
def show_img(img_name, img_src):
    cv2.imshow(img_name, img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("./images/receipt.jpg")
ration = img.shape[0] / 500.0
orig = img.copy()

img = myutils.resize(img, height = 500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

show_img("edged", edged)

# 寻找各个数字轮廓(轮廓是乱序的)
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
for c in cnts:
    peric = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peric, True)
    if (len(approx) == 4):
        screenCnt = approx
        break
res = cv2.drawContours(img , [screenCnt], -1, (0, 0, 255), 3)
show_img("res", res)


warped = myutils.four_point_transform(orig, screenCnt.reshape(4, 2) * ration)
print(warped.shape)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite("./images/scan.png", warped)
cv2.imshow("Scanned",myutils.resize(warped, height = 650))
cv2.waitKey(0)
