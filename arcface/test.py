from batchflow_hub.arcface import ArcFaceProcessor
import numpy as np

i = np.ones((120,120,3))

arcface = ArcFaceProcessor()
arcface.open()

o = arcface.process({"image":i, "detections":[[0,30,30,50],[40,30,60,50]],"filename":"test2.jpg"})
o = arcface.process({"image":i*2, "detections":[[0,30,30,50],[40,30,60,50]],"filename":"test2.jpg"})


arcface.batch_size=2
o = arcface.process_batch({"image":[i,i*2], "detections":[[[0,30,30,50],[40,30,60,50]],[[0,30,30,50],[40,30,60,50]]],"filename":["test.jpg","test2.jpg"]})


pass
