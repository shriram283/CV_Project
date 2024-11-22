import cv2
import numpy as np

def getContours(img, threshold=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, threshold[0], threshold[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThresh = cv2.erode(imgDial, kernel, iterations=2)
    
    if showCanny:
        cv2.imshow('Canny Image', imgThresh)
    
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")
    
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            boundingbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, boundingbox, i])
            else:
                finalContours.append([len(approx), area, approx, boundingbox, i])
    
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    print(f"Contours after filtering (minArea={minArea}, filter={filter}): {len(finalContours)}")
    
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    
    return img, finalContours

def reorder(points):
    newpoints = np.zeros_like(points)
    points = points.reshape((4, 2))
    add = points.sum(1)
    newpoints[0] = points[np.argmin(add)]
    newpoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newpoints[1] = points[np.argmin(diff)]
    newpoints[2] = points[np.argmax(diff)]
    return newpoints

def warpImg(img, points, width, height, pad=0):  # Set pad=0 to avoid cropping/zooming
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width, height))
    
    return imgWarp

def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5
