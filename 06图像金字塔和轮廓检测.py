import cv2
import numpy as np
import matplotlib.pyplot as plt



# 展示图像函数
def show_img(img_name, img_src):
    cv2.imshow(img_name, img_src)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


# 高斯金字塔，拉普拉斯金字塔

img = cv2.imread("AM.png")
show_img("AM", img)

# 向上采样，就是放大
up = cv2.pyrUp(img)
show_img("up_img", up)

# 向下采样，就是缩小
down = cv2.pyrDown(img)
show_img("down_img", down)

# 先向上采样，再向下采样，两次都存在精度损失，跟原始对比，效果肯定差
up_down = cv2.pyrDown(up)
res = np.hstack((img, up_down))
show_img("img-up_down", res)

# image代表输入的图片。注意输入的图片必须为二值图片。若输入的图片为彩色图片，必须先进行灰度化和二值化。
# mode
# 表示轮廓的检索模式，有4种：
# cv2.RETR_EXTERNAL
# 表示只检测外轮廓。
# cv2.RETR_LIST
# 检测的轮廓不建立等级关系。
# cv2.RETR_CCOMP
# 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一
# 个连通物体，这个物体的边界也在顶层。
# cv2.RETR_TREE
# 建立一个等级树结构的轮廓。
# method
# 为轮廓的近似办法，有4种：
# cv2.CHAIN_APPROX_NONE
# 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1 - x2），                       abs（y2 - y1）） <= 1。
# cv2.CHAIN_APPROX_SIMPLE
# 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个
# # 轮廓检测:findContours
img = cv2.imread("contours.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
show_img("binary", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
draw_img = img.copy()
# -1表示所有轮廓，2表示粗细，(0, 0, 255)表示BGR颜色
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
show_img("contours", res)

# 计算轮廓面积
print(cv2.contourArea(contours[0]))

# 计算周长
print(cv2.arcLength(contours[0], True))

# 模板匹配‘

img = cv2.imread("famouspeople.png", 0)
img2 = img.copy()
template = cv2.imread("template.png", 0)
h, w = template.shape[:2]


res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = min_loc
bottom_right = (top_left[0]+w, top_left[1]+h)
cv2.rectangle(img2, top_left, bottom_right, 255, 2)
plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.imshow(img, cmap="gray")
plt.title("face.jpg")
# 隐藏坐标轴
# plt.xticks([]), plt.yticks([])
plt.subplot(222)

plt.imshow(img2, cmap="gray")
plt.title("face in leano.jpg")
# 隐藏坐标轴
# plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(template, cmap="gray")
plt.title("template_face .jpg")
plt.show()


# 匹配到多个对象
iimg_rgb = cv2.imread("1.png")
img_copy = img_rgb.copy()
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread("2.png", 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0]+w, pt[1]+h)
    cv2.rectangle(img_copy, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow("img_result", img_copy)
cv2.waitKey(0)