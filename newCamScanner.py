import cv2 
import numpy as np

img=cv2.imread('photos/test1.jpg')


#resize images
resized_img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))

#change to gray 

gray = cv2.cvtColor(resized_img,cv2.COLOR_RGB2GRAY)

#bluring

blur = cv2.GaussianBlur(gray,(5,5),0)
# #canny 

canny = cv2.Canny(blur,30,50)

# #segmentation by threshholding

# threshholding = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)



#contours

contours,hiearchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

for contour in contours:

    
    epsilon = 0.02*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)

    if len(approx)==4:

        bast_contour = approx
        cv2.drawContours(canny, [approx], -1, (0, 255, 0), 3)


        break

#warping

points = np.float32(approx.reshape(4, 2))
    
flat_paper = np.float32([
            [0, 0],             
            [300, 0],        
            [300, 400],    
            [0, 400]        
        ])
    
M = cv2.getPerspectiveTransform(points, flat_paper)

warped = cv2.warpPerspective(resized_img, M, (300, 400))
    
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)

