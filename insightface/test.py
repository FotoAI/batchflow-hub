from batchflow_hub.insightface import InsightFace
import cv2

img1 = cv2.imread("/home/tushar/hdd/downloads/sample_photos/side/DSC_7665.JPG")
img2 = cv2.imread("/home/tushar/hdd/downloads/sample_photos/side/DSC_7668.JPG")

iface = InsightFace()
iface.open()
o = iface.process({"image": img1})
for i, (det, crop) in enumerate(zip(o["detections"], o["face_crops"])):
    x1, y1, x2, y2 = det["face"][:4]
    im = cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for k, (x, y) in det["landmarks"].items():
        im = cv2.circle(im, (int(x), int(y)), 7, (0, 255, 0), -1)

    cv2.imwrite(f"img1_{i}.jpg", crop.astype("uint8"))
cv2.imwrite("img1.jpg", im)

# fd.batch_size=2
# o = fd.process_batch({"image":[img1],"filename":["img1"]})
# for dets,name in zip(o["detections"],o["filename"]):
#     for i,det in enumerate(dets):
#         x1,y1,x2,y2 = det["face"][:4]
#         im = cv2.rectangle(img1,(x1,y1),(x2,y2), (255,0,0),3)
#         for k,(x,y) in det["landmarks"].items():

#             im = cv2.circle(im, (int(x),int(y)), 7, (0,255,0),-1)

#         cv2.imwrite(f"{name}_b_{i}.jpg",im[y1:y2,x1:x2])
#     cv2.imwrite(f"{name}_b.jpg", im)
