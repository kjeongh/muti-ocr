
import cv2
import numpy as np
from PIL import Image


# s3에 저장된 이미지를 가져와서 전처리한 뒤 텍스트 추출


def grouping_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #흑백 변환
    (H, W) = gray.shape

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 21))

    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    close_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    close_thresh = cv2.erode(close_thresh, None, iterations=2)

    cv2.imwrite('output_image.png', close_thresh)


