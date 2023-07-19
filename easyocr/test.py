import easyocr
import cv2
import time

reader = easyocr.Reader(
    ["en"]
)  # this needs to run only once to load the model into memory


img = cv2.imread("../yolov7/o0.jpg")
s = time.time()
result = reader.readtext(img, threshold=0.7, text_threshold=0.8)
print(time.time() - s)
print(result)
