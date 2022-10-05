import yoloTool
import cv2

img = cv2.imread("./1.png", cv2.IMREAD_COLOR)
imgYolo = yoloTool.detect(img)

cv2.imshow('detection', imgYolo)
cv2.waitKey(0)