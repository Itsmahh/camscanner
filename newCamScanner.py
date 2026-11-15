import cv2 
import numpy as np

img=cv2.imread('photos/test3.jpg')


#resize images
resized_img = cv2.resize(img,(300,400))

#change to gray 

gray = cv2.cvtColor(resized_img,cv2.COLOR_RGB2GRAY)

#bluring

# blur = cv2.GaussianBlur(gray,(5,5),1.0)

blur = cv2.bilateralFilter(gray,5,150,150)

# #canny 

canny = cv2.Canny(blur,200,75)

# dilated_edges = cv2.dilate(canny, (7,7), iterations=10)

# #segmentation by threshholding

# threshholding = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)



#contours

contours,hiearchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

best_contour = None

for contour in contours:

    
    epsilon = 0.02*cv2.arcLength(contour,True)

    approx = cv2.approxPolyDP(contour,epsilon,True)

    if len(approx)==4 and cv2.contourArea(contour)>1000:

        best_contour = approx
        image_copy = resized_img.copy()
        cv2.drawContours(image_copy, [approx], -1, (0, 255, 0), 3)


        break
    else:
        print ('contour not found')


#warping

# def reorder_points(points):
points = approx.reshape(4, 2)

center = np.mean(points, axis=0)


for point in points:
    x = point[0]
    y = point[1]
    if x < center[0] and y < center[1] :
        top_left = point
        print(top_left)
    elif x > center[0] and y > center[1] :
        bottom_right = point
        print(bottom_right)

    elif x > center[0] and y < center[1] :
        top_right = point
        print(top_right)

    elif x < center[0] and y > center[1] :
        bottom_left = point
        print(bottom_left)
    else:
        print("these coordinates have been repeated.")

order = np.array([top_left,top_right,bottom_right,bottom_left],dtype = np.float32)


    
paper = np.float32([
            [0, 0],             
            [300, 0],        
            [300, 400],    
            [0, 400]        
        ])
    
M = cv2.getPerspectiveTransform(order, paper)

warped = cv2.warpPerspective(resized_img, M, (300, 400))




warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped_blur = cv2. GaussianBlur(warped_gray,(5,5),1.0)
threshholding = cv2.adaptiveThreshold(warped_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
# thicker_text = cv2.erode(threshholding, (5,5), iterations=5)

cv2.imshow("resized image", resized_img)
cv2.imshow("gray", gray)
cv2.imshow("blur", blur)
cv2.imshow("canny", canny)
cv2.imshow("image copy", image_copy)
cv2.imshow("warped", warped)
cv2.imshow("warped", threshholding)


cv2.waitKey(0)

