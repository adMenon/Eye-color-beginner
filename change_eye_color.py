#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 12:51:59 2018

@author: adityamenon
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#for detecting the eyes in an image we use the haarcascade classifier for eyes
#this is an inbuilt feature of opencv
#for convenience the classifier has been attached to the file
# make sure the correct full location is given for the image and also the classifier
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#print(face_cascade.empty())

#make sure the full location is given for the image file
img = cv2.imread('index_face.jpg_attachment_918217.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(gray)

#for storing the details of the detected eyes
eye=[]
for (ex,ey,ew,eh) in eyes:
    #draw boxes around eyes
    #cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    eye.append(gray[ey:ey+eh,ex:ex+ew])
eye = np.array(eye)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
#applying histogram equalisation on the images of the eyes
#this will help us in detecting circles using hough transform
equ1 = cv2.equalizeHist(eye[0])
equ2 = cv2.equalizeHist(eye[1])

#applying hough transform to find the circles
circle1 = cv2.HoughCircles(equ1,cv2.HOUGH_GRADIENT,1,20,
                             param1=50,param2=30,minRadius=0,maxRadius=0)
circle2 = cv2.HoughCircles(equ2,cv2.HOUGH_GRADIENT,1,20,
                             param1=50,param2=30,minRadius=0,maxRadius=0)

circle1 = np.uint16(np.around(circle1))
circle2 = np.uint16(np.around(circle2))

# the following part of the code deals with finding the non pupil part of the iris
# and saving the corresponding locations 
# the entire eyeball can be colored by removing the lower limit in the first if condition
x = circle1[0][0][0]-0.5
y = circle1[0][0][1]+2
r = min(circle1[0][0][2],10)-1.7
l1 = []
for i in range(eye[0].shape[0]):
    for j in range(eye[0].shape[1]):
        if (x-j)**2 + (y-i)**2 < r**2:   
            if  equ1[i][j] < 100:
                eye[0][i][j]=0
                l1.append([i,j])
                
x = circle2[0][0][0]-2.5
y = circle2[0][0][1]+1.2
r = min(circle2[0][0][2],10)-2.2
l2 = []
for i in range(eye[1].shape[0]):
    for j in range(eye[1].shape[1]):
        if (x-j)**2 + (y-i)**2 < r**2:   
            if  0<equ2[i][j] < 140:
                eye[1][i][j]=0
                l2.append([i,j])

# the following code deals with coloring the eyes
ex,ey,ew,eh = eyes[0]
#print(ex,ey)
for i in l1:
    if np.mean(img[ey+i[0]][ex+i[1]]) <180:
        #color input should be of order BGR
        img[ey+i[0]][ex+i[1]] = np.array([50,0,70]) #a shade of red color is used to make the image a bit realistic 
ex,ey,ew,eh = eyes[1]
#print(ex,ey)
for i in l2:
    if np.mean(img[ey+i[0]][ex+i[1]]) <180:
        #color input should be of order BGR
        img[ey+i[0]][ex+i[1]] = np.array([50,0,70]) # same color as above 
        
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite('eye_color_changed.jpg',img)
plt.show()