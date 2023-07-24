from batchflow_hub.hsemotion import HSEmotion
from batchflow_hub.insightface import InsightFace
import cv2


model_source_if = [
                {
                    "source": "backblaze",
                    "filename": "det_10g.onnx",
                    "key": "models/insightface/det_10g.onnx",
                    "bucket_name": "fotoowl-master-bucket",
                },
                {
                    "source": "backblaze",
                    "filename": "w600k_r50.onnx",
                    "key": "models/insightface/w600k_r50.onnx",
                    "bucket_name": "fotoowl-master-bucket",
                },
                {
                    "source": "backblaze",
                    "filename": "2d106det.onnx",
                    "key": "models/insightface/2d106det.onnx",
                    "bucket_name": "fotoowl-master-bucket",
                },
            ]

insight_face = InsightFace(model_source=model_source_if)
insight_face.open()


model_source = {
    "source": "backblaze",
    "filename": "enet_b0_8_best_vgaf.pt",
    "key": "models/hsemotion/enet_b0_8_best_vgaf.pt",
    "bucket_name": "fotoowl-master-bucket",
}
hsemotion = HSEmotion(model_path="enet_b0_8_best_vgaf.pt", model_source=None)
hsemotion.open()




img1 = cv2.imread("../imgs/img1.jpg")
img2 = cv2.imread("../imgs/img2.jpg")


ctx_batch = []

for img in [img1, img2]:
    out = insight_face.process({"image":img})
    face_bbox = out["detections"][0]["face"]
    ctx_batch.append({**out, "face_bbox":face_bbox})


ctx_batch = hsemotion.process_batch(ctx_batch)
print(ctx_batch)




# o = yolo.process({"image": img})
# human_detection = o["human_detection"]

# for i_human_detection in human_detection:
#     for i, bbox in enumerate(i_human_detection):
#         x1, y1, x2, y2 = map(int, bbox[:4])
#         print(x1, y1, x2, y2)
#         crop = img[y1:y2, x1:x2]
#         cv2.imwrite(f"o{i}.jpg", crop)
