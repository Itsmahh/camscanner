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

# canny = cv2.Canny(blur,30,50)

# #segmentation by threshholding

threshholding = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)



#contours

contours,hiearchy = cv2.findContours(threshholding,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# s_list=[]
# c_list=[]

for contour in contours:
    max_area  = 0

    area = cv2.contourArea(contour)

    if area >1000:
        continue

    epsilon = 0.02*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)

    if len(approx)==4:

        bast_contour = contour
        cv2.drawContours(threshholding, contour, -1, (0, 255, 0), 3)

        cv2.imshow('Contours', threshholding)






    #warping 

M1 = np.float32(approx.reshape(4, 2))
M2 = np.float32([
    [0, 0],
    [300, 0],
    [300, 400],
    [0, 400]
])
M= cv2.getAffineTransform(M1,M2)

warping = cv2.warpAffine(img,M,(300,400))
cv2.imshow("warping image",warping)


    # for contour in new_contours:

    #     biggest_area = cv2.contourArea(contour)

    #     if biggest_area > 1000:
    #         print(contour)

        # s_list.append(s)
        # c_list.append(contour)

# if s_list:
#     max_s = max(s_list)
#     index = s_list.index(max_s)
#     best_counter = c_list[index]
    # approx = cv2.drawContours(canny, best_counter, -1, (0,255,0), 3)
    # cv2.imshow("adsd",approx)
cv2.waitKey(0)