import cv2
import imutils
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist
import skimage




img = cv2.imread('D:/code/_line_detection/T4_Y504/7/5.jpg')
image = img
img = cv2.medianBlur(img, 7)
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('gray2.jpg', gray2 )

h, s, v = cv2.split(gray2)
cv2.imwrite('V.jpg', v )

ret, binary = cv2.threshold(v, 50, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
cv2.imwrite('threshold2.jpg', binary )






# 读入图片
#img = cv2.imread('D:/code/0331_2/img316.jpg')
# 中值滤波，去噪
#img = cv2.medianBlur(img, 3)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
#cv2.imshow('original', gray)

# 阈值分割得到二值化图片
#ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
#ret, binary = cv2.threshold(gray, 2, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

# 膨胀操作
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
bin_clo = cv2.erode(binary, kernel2, iterations=10)
cv2.imwrite('bin_clo.jpg', bin_clo )

# 连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)

# 查看各个返回值
# 连通域数量
print('num_labels = ',num_labels)


biggest = 0
size = 0


for i in range(1, num_labels):
        
        if int(stats[i][4]) > size:
                size = int(stats[i][4])
                biggest = i


# 不同的连通域赋予不同的颜色
#output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#mask = labels == biggest
#output[:, :, 0][mask] = np.random.randint(0, 255)
#output[:, :, 1][mask] = np.random.randint(0, 255)
#output[:, :, 2][mask] = np.random.randint(0, 255)

#for i in range(1, num_labels):
#    print(i)

#    mask = labels == i
#    output[:, :, 0][mask] = np.random.randint(0, 255)
#    output[:, :, 1][mask] = np.random.randint(0, 255)
#    output[:, :, 2][mask] = np.random.randint(0, 255)



print("biggest is ", biggest)

print("biggest area is ", stats[biggest][4])

#cv2.imwrite('oginal.jpg', output )



b = np.random.randint(0, 256)
g = np.random.randint(0, 256)
r = np.random.randint(0, 256)

#for row in range(img.shape[0]):
#    for col in range(img.shape[1]):
#        if (labels[row, col] == biggest):
#                image[row, col] = (255,245,0)

width = 0
array = []
for row in range(img.shape[0]):
    count = 0
    for col in range(img.shape[1]):
        if (labels[row, col] == biggest):
                count = count+1
    if (count > img.shape[1]*0.8):
        array.append(row)
        width += 1
        #for col in range( int(img.shape[1]*0.5) ):
                #image[row, col] = (255,245,0)

                
start_point = (0, array[0])
end_point = (img.shape[1], array[0])
color = (0, 255, 0) # green
thickness = 3 # 寬度
cv2.line(image, start_point, end_point, color, thickness)


start_point = (0, array[0]+width)
end_point = (img.shape[1], array[0]+width)
color = (0, 255, 0) # green
thickness = 3 # 寬度
cv2.line(image, start_point, end_point, color, thickness)

    
cv2.imwrite('image.jpg', image )
print("width = ", width)
cv2.waitKey()
