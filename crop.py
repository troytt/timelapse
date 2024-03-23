import cv2

# 打开照片文件
img = cv2.imread("/Users/taodao/Downloads/timelapse/2023.04.13_18.24.41/Proj96_img00000001.jpg")

# 获取图像的宽度和高度
height, width, channels = img.shape
size = min(width, height)

# 计算裁剪后图像的左上角和右下角坐标
x = (width - size) // 2
y = (height - size) // 2
w = size
h = size

# 使用numpy数组切片将图像裁剪为正方形
square_img = img[y:y+h, x:x+w]

# 保存裁剪后的图像为新文件
cv2.imwrite("/Users/taodao/Downloads/timelapse/square_image.jpg", square_img)
