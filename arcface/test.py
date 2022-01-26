from batchflow_hub.arcface import ArcFace
import numpy as np

i = np.ones((120, 120, 3))

arcface = ArcFace()
arcface.open()

o = arcface.process(
    {
        "image": i,
        "face_crops": [i, i],
        "filename": "test2.jpg",
    }
)
o = arcface.process(
    {
        "image": i * 2,
        "face_crops": [i, i],
        "filename": "test2.jpg",
    }
)


arcface.batch_size = 2
o = arcface.process_batch(
    {
        "image": [i, i * 2],
        "face_crops": [[i, i], [i, i]],
        "filename": ["test.jpg", "test2.jpg"],
    }
)
