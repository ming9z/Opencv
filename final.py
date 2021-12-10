#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np

# 변수들 초기화.
hsv = 0
lower_color1 = 0
upper_color1 = 0
lower_color2 = 0
upper_color2 = 0
lower_color3 = 0
upper_color3 = 0


# 트랙바 생성시 필요
def nothing(x):
    pass


# 마우스 클릭 이벤트
def mouse_callback(event, x, y, flags, param):
    global hsv, lower_color1, upper_color1, lower_color2, upper_color2, lower_color3, upper_color3, threshold

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환 (x, y 값으로 저장)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv2.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        threshold = cv2.getTrackbarPos('threshold', 'img_result')  # 현재 threshold 값을 가져옴.

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 픽셀값의 범위를 정함
        
        # 첫번째 경우 hsv의 h 값이 10 이하 (빨간색 ?)
        if hsv[0] < 10:
            print("case1")
            lower_color1 = np.array([hsv[0] - 10 + 180, threshold, threshold])  # 색상만 조절
            upper_color1 = np.array([180, 255, 255])
            lower_color2 = np.array([0, threshold, threshold])
            upper_color2 = np.array([hsv[0], 255, 255])
            lower_color3 = np.array([hsv[0], threshold, threshold])
            upper_color3 = np.array([hsv[0] + 10, 255, 255])


            # hsv 의 h값 170이상 ( 파란색 ?)
        elif hsv[0] > 170:
            print("case2")
            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([180, 255, 255])
            lower_color2 = np.array([0, threshold, threshold])
            upper_color2 = np.array([hsv[0] + 10 - 180, 255, 255])
            lower_color3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_color3 = np.array([hsv[0], 255, 255])

            # 나머지
        else:
            print("case3")
            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([hsv[0] + 10, 255, 255])
            lower_color2 = np.array([hsv[0] - 10, threshold, threshold])
            upper_color2 = np.array([hsv[0], 255, 255])
            lower_color3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_color3 = np.array([hsv[0], 255, 255])
'''
        print(hsv[0])
        print("@1", lower_color1, "~", upper_color1)
        print("@2", lower_color2, "~", upper_color2)
        print("@3", lower_color3, "~", upper_color3)

'''

cv2.namedWindow('img_color')
cv2.setMouseCallback('img_color', mouse_callback)

cv2.namedWindow('img_result')
cv2.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'img_result', 30)  # 트랙바의 초깃값 30

# 웹캠과 연결.
cap = cv2.VideoCapture(0)

while (True):

    # 캠에 연결한 cap을 읽어옴.
    ret, img_color = cap.read()
    height, width = img_color.shape[:2]
    
    # 원본 이미지를 resize함.
    img_color = cv2.resize(img_color, (width, height), interpolation=cv2.INTER_AREA)

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    img_mask1 = cv2.inRange(img_hsv, lower_color1, upper_color1)
    img_mask2 = cv2.inRange(img_hsv, lower_color2, upper_color2)
    img_mask3 = cv2.inRange(img_hsv, lower_color3, upper_color3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # morphology 연산으로 노이즈 ( 점, 구멍) 제거 / 조명이 너무 쌔면 나옴.
    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

    # result에 마스킹한걸 비트연산.
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    # 라벨을 붙일 이미지, 라벨 지정된 객체 , 라벨 갯수? , 라벨 중심 
    # (라벨) 물체 영역의 자표를 얻음 / 중심 크기 / 외곽.
    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(img_mask)
    # 쉽게 라벨링 할 수 있음

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        # connectcd부터 반환받은 값으로 물체 주변의 외곽을 그리기 위해 좌표를 받음 , 물체 중심 좌표 얻음.
        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        # 노이즈 물체의 영역비 50 이면 중심좌표 빨간원 
        # 라벨로 지정된 물체 중심에 작은 원을 그림. 
        if area > 50:
            cv2.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 5)
            cv2.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255))

    cv2.imshow('img_color', img_color)
    cv2.imshow('img_mask', img_mask)
    cv2.imshow('img_result', img_result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()


# In[ ]:




