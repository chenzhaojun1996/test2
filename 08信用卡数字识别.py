import cv2
import matplotlib.pyplot  as plt
import numpy as np
import myutils


# 使用cv.read展示图像函数
def show_img(img_name, img_src):
    cv2.imshow(img_name, img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


template = cv2.imread("./images/ocr_a_reference.png")
#  灰度图
ref = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# 二值化
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
show_img("template_threshold:", ref)
print(ref.shape)

# 寻找各个数字轮廓(轮廓是乱序的)
contours, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#  画出展示所有数字轮廓
res = cv2.drawContours(template.copy(), contours, -1, (0, 0, 255), 3)
show_img("contours", res)
# 调用myutils文件对contours进行排序(从左到右)，第一个维度代表是轮廓返回
refCnts = myutils.sort_contours(contours, method="left-to-right")[0]
# 存放模板的数字ROI
digits = {}

for (i, c) in  enumerate(refCnts):
    # cv2.imread()返回的是(height, width, channels),
    #x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    # 但是ref是hxwxc，所以提取模板数字roi为如下：
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

# 初始卷积核，定义开闭操作的卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


# 读取银行卡，首先resize，灰度化
img = cv2.imread("./images/credit_card_01.png")
img = myutils.resize(img, width=300)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show_img("gray:", gray)

# 礼帽操作：原始输入-开运算
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# 单独x方向，将梯度变大
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# 绝对化梯度x
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# 归一化操作
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
# 变为uint8类型
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

thresold = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]


# 闭操作
gradX = cv2.morphologyEx(thresold, cv2.MORPH_CLOSE, sqKernel)

# 得出闭操作后的轮廓

threCnts, hierarchy = cv2.findContours(gradX.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threCnts
cur_img = img.copy()
# 画轮廓
res = cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
show_img("locate digts ：", res)
# 放4个数组区域
locs = []

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    # 根据长宽比来废除银行卡其他区域
    ar = w / float(h)
    if ar > 2.5 and ar <  4.0 :
        if(w > 40 and w < 55) and (h >10 and h < 20):
            # 将银行卡图片数字区域块存放
            locs.append((x, y, w, h))
# 数字区域升序排放，以第一个维度x来指标
locs = sorted(locs, key=lambda x:x[0])

outputs = []
# 大区域数字遍历
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # 数字区域输出
    groupOutput = []

    group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]
    # show_img("group", group)
    # 在只有两种颜色情侣下，0代表自动查找阈值
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    groupCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 每一个大块数字区域有4个数字，这4个数字得排序一次
    digitCnts = myutils.sort_contours(groupCnts, method="left-to-right")[0]
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        # 找出单独数字并且seize
        roi = cv2.resize(roi, (57, 88))
        # show_img("roi",roi)
        # 4个数字的具体得分
        scores = []
        # digits是模板数字
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            # minval， maxval，minloc，maxloc
            # 把maxval当成匹配，因为TM_CCOEFF是越相关，系数越大
            (_, score, _, _)  = cv2.minMaxLoc(result)
            scores.append(score)
        # 将一个数字与10个模板数字匹配，取最大值
        groupOutput.append(str(np.argmax(scores)))
    print(groupOutput)
    # 画框，左上角x，y，右下角x，y
    cv2.rectangle(img, (gX-5, gY-5), (gX+gW+5, gY+gH+5), (0, 0, 255), 2)

    cv2.putText(img, "".join(groupOutput), (gX, gY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 255),2)
    outputs.extend(groupOutput)
show_img("img",img)














