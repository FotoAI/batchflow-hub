# RetinaFace 

### Face Detector 
code inspired from deepface retinaface


## Installation

1. Cloning repo
```
git clone https://github.com/FotoAI/batchflow-hub.git
cd insightface
pip install .
```

2. Uing pip+https
```
pip install git+https://github.com/FotoAI/batchflow-hub.git#subdirectory=insightface
```

## How to Use

```
from batchflow_hub.insightface import InsightFace
import cv2

# create model instance
iface = InsightFace()

# load model in memory
iface.open()

# test image
img = cv2.imread("path to your image")
# BGR to RGB
img = img[...,::-1]

# input signature
ctx={"image":img}

# process image
output = iface.process(ctx)

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
        }],
    "encodings":[...],
    }

```
