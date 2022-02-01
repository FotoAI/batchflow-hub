# ArcFace

### Face Encoder for Face Recognition

code inspired from deepface arcface

## Installation

1. Cloning repo

```
git clone https://github.com/FotoAI/batchflow-hub.git
cd arcface
pip install .
```

2. Uing pip+https

```
pip install git+https://github.com/FotoAI/batchflow-hub.git#subdirectory=arcface
```

## How to Use

```
from batchflow_hub.arcface import ArcFace
import cv2

# create model instance
encoder = ArcFace(alignface=False)

# load model in memory
encoder.open()

# test image
img = cv2.imread("path to your image")
# BGR to RGB
img = img[...,::-1]

# input signature
ctx={"image":img, "detections": [{"face":[...]}]}

# process image
output = encoder.process(ctx)

# output signature
print(output)
>> {"image": np.array(...),
    "detections":
        {"face": [...]},
    "encodings":[...]
    }

```
