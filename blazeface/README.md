# RetinaFace 

### Face Detector 
Taken from mediapipe blazeface


## Installation

1. Cloning repo
```
git clone https://github.com/FotoAI/batchflow-hub.git
cd blazeface
pip install .
```

2. Uing pip+https
```
pip install git+https://github.com/FotoAI/batchflow-hub.git#subdirectory=blazeface
```

## How to Use

```
from batchflow_hub.blazeface import BlazeFace
import cv2

# create model instance
detector = BlazeFace()

# load model in memory
detector.open()

# test image
img = cv2.imread("path to your image")
# BGR to RGB
img = img[...,::-1]

# input signature
ctx={"image":img}

# process image
output = detector.process(ctx)

# output signature
print(output)
>> {"image": np.array(...), 
    "detections":
        [{"face": [x1,y1,x2,y2],
            "landmarks":
                "right_eye":[x,y],
                "left_eye": [x,y],
                "nose": [x,y],
                "mouth" : [x,y]
        }]
    }

```
