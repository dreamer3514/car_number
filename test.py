from os import listdir
import operator
import numpy as np
import cv2

simage = cv2.imread("img/1.jpg",0);

s = cv2.imread("img/1.jpg")
colorImage = s.copy()
hsv=cv2.cvtColor(s,cv2.COLOR_BGR2HSV)
sum=0
for i in range(simage.shape[0]):
    for j in range(simage.shape[1]):
       sum+=simage[i][j]
th = sum/(simage.shape[0]*simage.shape[1])
gaussian = cv2.GaussianBlur(simage, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
ret,th1 = cv2.threshold(gaussian,th-10,255,cv2.THRESH_BINARY)
canny = cv2.Canny(th1,80, 120)

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(25, 10))
iClose = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel1)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(25, 10))
open1 = cv2.morphologyEx(iClose, cv2.MORPH_OPEN, kernel2)

kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 22))
open2 = cv2.morphologyEx(open1, cv2.MORPH_OPEN, kernel3)

binary,contours,hierarchy  = cv2.findContours(open2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(contours, key=cv2.contourArea, reverse=True)

for re in cnt:
    x, y, w, h = cv2.boundingRect(re)
    cropImg = hsv[y:y + h, x:x + w]
    total = w*h
    count = 0
    if(w/h>2):
        for i in range(cropImg.shape[0]):
             for j in range(cropImg.shape[1]):
                if((cropImg[i][j][0]>100 and cropImg[i][j][0]<124 )):
                     count+=1
    if(count>total*0.5):
        result = s[y:y + h, x:x + w]
    cv2.rectangle(s, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("hsv",hsv)
cv2.imshow("open2",open2)
cv2.imshow("a",result)
cv2.imshow("b",s)
cv2.waitKey(0)