from batchflow_hub.retinaface import RetinaFace
import cv2

im = cv2.imread("imgs/img1.jpg")
fd = RetinaFace()
fd.open()
o = fd.process({"image":im})
for det in o["detections"]:
    x1,y1,x2,y2 = det["face"][:4]
    im = cv2.rectangle(im,(x1,y1),(x2,y2), (255,0,0),3)
    for k,(x,y) in det["landmarks"].items():

        im = cv2.circle(im, (int(x),int(y)), 7, (0,255,0),-1)
cv2.imwrite("im.jpg", im)

# fd.batch_size=2
# fd.process_batch({"image":[im, im]})
# pass