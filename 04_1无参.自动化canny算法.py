# -*- coding: utf-8 -*-
# 导入一些python包
import numpy as np
import argparse
import glob
import cv2
import os


# 定义auto_canny函数
def auto_canny(image, sigma=0.33):
    # 计算单通道像素强度的中位数
    v = np.median(image)

    # 选择合适的lower和upper值，然后应用它们
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


# 设置一些需要修改的参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "dst_img", required=True, help="dst_img")
args = vars(ap.parse_args())

# 创建可视化文件夹
file_dir = "vis/"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

# 遍历文件夹中的每一张图片
i = 0
img_names = glob.glob(args["images"] + "/*.jpg")
for imagePath in img_names:
    # 读取图片
    image = cv2.imread(imagePath)
    # 灰度化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 分别使用宽阈值、窄阈值和自动确定的阈值进行测试
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)
    result = np.hstack([wide, tight, auto])
    i += 1

    save_name = "vis/" + str(i) + ".png"
    # 显示并保存结果
    cv2.imshow("Original", image)
    cv2.imshow("Edges", result)
    cv2.imwrite(save_name, result)
    cv2.waitKey(0)
