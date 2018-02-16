from os import listdir
import operator
import numpy as np
import cv2
import matplotlib.pyplot as plt

s = cv2.imread("img/5-1.jpg",0)
# gaussian = cv2.GaussianBlur(s, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 1))
# iClose = cv2.morphologyEx(s, cv2.MORPH_CLOSE, kernel1)
#
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(25, 11))
# open1 = cv2.morphologyEx(iClose, cv2.MORPH_OPEN, kernel2)

#逐列扫描通过像素点分割车牌
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))
open2 = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel3)

ret,th1 =cv2.threshold(open2,127,255,cv2.THRESH_BINARY)
# print(th1)
cv2.imshow("d",th1)


a=range(th1.shape[1]-1)

b=[]
for i in range(th1.shape[1]-1):
    count = 0
    for j in range(th1.shape[0]-1):
        if th1[j][i]==255:
            count=count+1
    b.append(count)
plt.plot(a,b)
plt.show()
start=0
end = 0
img=[]
flag=-1
i=0
print(b)
for index in range(th1.shape[1]-1):
    if(b[index]<3 and flag==1):
        end = index
        crop = s[0:th1.shape[0] - 1, start:end]
        # cv2.imshow(str(i), crop)
        img.append(crop)
        flag=flag*-1
    if(b[index]>3 and flag==-1):
        start=index
        flag=1
        print(start)
d = s[0:th1.shape[0] - 1, start:th1.shape[1]-1]
# cv2.imshow("1", d)
img.append(d)
# print(img)
for each in img:
    cv2.imshow(str(i),each)
    i=i+1

cv2.imwrite("F://number.jpg",img[5])
cv2.imshow("crop",crop)
cv2.waitKey(0)