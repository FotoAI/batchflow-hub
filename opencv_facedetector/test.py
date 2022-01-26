from batchflow_hub.opencv_facedetector import OpenCVFaceDetector

import cv2

im = cv2.imread("imgs/img1.jpg")
fd = OpenCVFaceDetector()
fd.open()
o = fd.process({"image":im})
for det in o["detections"]:
    x1,y1,x2,y2 = det["face"][:4]
    im = cv2.rectangle(im,(x1,y1),(x2,y2), (255,0,0),3)
    for k,(x,y) in det["landmarks"].items():

        im = cv2.circle(im, (int(x),int(y)), 7, (0,255,0),-1)
cv2.imwrite("im.jpg", im)

im2 = cv2.imread("imgs/img2.jpg")
images = [im, im2]
fd.batch_size=2
o = fd.process_batch({"image":images})

for i, (im, idet) in enumerate(zip(images, o["detections"])):
    for det in idet:
        x1,y1,x2,y2 = det["face"][:4]
        im = cv2.rectangle(im,(x1,y1),(x2,y2), (255,0,0),3)
        for k,(x,y) in det["landmarks"].items():
            im = cv2.circle(im, (int(x),int(y)), 7, (0,255,0),-1)
    cv2.imwrite(f"im{i}.jpg", im)


