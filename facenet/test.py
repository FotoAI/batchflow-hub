from batchflow_hub.facenet import FaceNet
import numpy as np
import cv2

img1 = "/home/tushar/hdd/My/FotoAI/ml/testing/gund_arc_no_batch/images/Frances/0.jpg"
img2 = "/home/tushar/hdd/My/FotoAI/ml/testing/gund_arc_no_batch/images/Frances/1.jpg"

img1 = cv2.imread(img1)[...,::-1]
img2 = cv2.imread(img2)[...,::-1]

arcface = FaceNet(variant=512)
arcface.open()

# o = arcface.process(
#     {
#         "image": img1,
#         "face_crops": [img1],
#         "filename": "img1.jpg",
#     }
# )
# o = arcface.process(
#     {
#         "image": img2 ,
#         "face_crops": [img2],
#         "filename": "img2.jpg",
#     }
# )


arcface.batch_size = 2
o = arcface.process_batch(
    {
        "image": [[img1,img2]],
        "face_crops": [[img1], [img2]],
        "filename": ["0.jpg", "1.jpg"],
    }
)
