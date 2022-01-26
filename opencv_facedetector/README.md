# RetinaFace 

### Face Detector 
code inspired from deepface retinaface


## Installation

1. Cloning repo
```
git clone https://github.com/FotoAI/batchflow-hub.git
cd retinaface
pip install .
```

2. Uing pip+https
```
pip install git+https://github.com/FotoAI/batchflow-hub.git#subdirectory=retinaface
```

## How to Use

```
from batchflow_hub.retinaface import RetinaFace
import cv2

# create model instance
retinaface_detector = RetinaFace()

# load model in memory
retinaface_detector.open()

# test image
img = cv2.imread("path to your image")
# BGR to RGB
img = img[...,::-1]

# input signature
ctx={"image":img}

# process image
output = retinaface_detector.process(ctx)

# output signature
print(output)
>> {"image": np.array(...), 
    "detections":
        [{"face": [x1,y1,x2,y2],
            "landmarks":
                "right_eye":[x,y],
                "left_eye": [x,y],
                "nose": [x,y],
                "mouth_right": [x,y],
                "mouth_left" : [x,y]
        }]
    }

```
