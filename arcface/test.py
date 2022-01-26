from batchflow_hub.arcface import ArcFace
import numpy as np

i = np.ones((120, 120, 3))

arcface = ArcFace(alignface=False)
arcface.open()

o = arcface.process(
    {
        "image": i,
        "detections": [{"face": [0, 30, 30, 50]}, {"face": [40, 30, 60, 50]}],
        "filename": "test2.jpg",
    }
)
o = arcface.process(
    {
        "image": i * 2,
        "detections": [{"face": [0, 30, 30, 50]}, {"face": [40, 30, 60, 50]}],
        "filename": "test2.jpg",
    }
)


arcface.batch_size = 2
o = arcface.process_batch(
    {
        "image": [i, i * 2],
        "detections": [
            [{"face": [0, 30, 30, 50]}, {"face": [40, 30, 60, 50]}],
            [{"face": [0, 30, 30, 50]}, {"face": [40, 30, 60, 50]}],
        ],
        "filename": ["test.jpg", "test2.jpg"],
    }
)
