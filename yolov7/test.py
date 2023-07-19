from batchflow_hub.yolov7 import YOLOv7
import cv2


model_source = {
    "source": "backblaze",
    "filename": "yolov7.pt",
    "key": "models/yolov7/yolov7.pt",
    "bucket_name": "fotoowl-master-bucket",
}
yolo = YOLOv7(model_path="yolov7.pt", model_source=None)
yolo.open()

img = cv2.imread("../imgs/marathon3.webp")
print(yolo)
o = yolo.process({"image": img})
human_detection = o["human_detection"]

for i_human_detection in human_detection:
    for i, bbox in enumerate(i_human_detection):
        x1, y1, x2, y2 = map(int, bbox[:4])
        print(x1, y1, x2, y2)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"o{i}.jpg", crop)
