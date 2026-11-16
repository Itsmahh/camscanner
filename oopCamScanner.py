import cv2 
import numpy as np

class CamScanner:
    def __init__(self,image_address):
        self.image = cv2.imread(image_address)
        self.resize = cv2.resize(self.image,(300,400),interpolation = cv2.INTER_AREA)
        
    def order_points(self,points):
        points = points.reshape(4, 2)

        center = np.mean(points, axis=0)


        for point in points:
            x = point[0]
            y = point[1]
            if x < center[0] and y < center[1] :
                top_left = point

            elif x > center[0] and y > center[1] :
                bottom_right = point

            elif x > center[0] and y < center[1] :
                top_right = point

            elif x < center[0] and y > center[1] :
                bottom_left = point


        order_points = np.array([top_left,top_right,bottom_right,bottom_left],dtype = np.float32)
        return order_points

    def scan_image(self):
        gray = cv2.cvtColor(self.resize,cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        canny = cv2.Canny(blur, 40, 120)
        dilated_edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,  np.ones((5, 5), np.uint8), iterations=2)

        contours,hiearchy = cv2.findContours(dilated_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        best_contour = None

        for contour in contours:
            approx = cv2.approxPolyDP(contour,0.02*cv2.arcLength(contour,True),True)
            if len(approx)==4 and cv2.contourArea(contour)>500 :
                best_contour = approx
                break
            else:
                print("contour not found")
        order = self.order_points(best_contour)
        paper = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])
        M = cv2.getPerspectiveTransform(order, paper)
        warped = cv2.warpPerspective(self.resize, M, (300, 400))





test_images = ['photos/test1.jpg', 'photos/test2.jpg', 'photos/test3.jpg', 'photos/test4.jpg']

for img_address in test_images:
    scanner = CamScanner(img_address)
    scanned = scanner.scan_image()

    if scanned is not None:
        cv2.imshow(f"Result: {img_address}", scanned)
        cv2.waitKey(0)
    
