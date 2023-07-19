from typing import Dict, Tuple, Union
from batchflow.processors.core import ModelProcessor
from typing import List, Any
import numpy as np
import cv2
from batchflow.decorators import log_time
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from batchflow.storage import get_storage
import os
from batchflow.constants import BATCHFLOW_HOME
import torch
from torchvision import transforms
from PIL import Image


class EfficientNet(ModelProcessor):
    def __init__(
        self,
        root: str = None,
        name: str = "b0",
        model_source: Dict[str, str] = None,
        target_size=(160, 160),
        *args,
        **kwargs,
    ) -> None:
        # inject default model_source if model_path and model_source is None

        self.model = None
        self.name = name
        self.activation = None
        self.tfms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def open(self):
        self.model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            f"nvidia_efficientnet_{self.name}",
            pretrained=True,
        )
        self.model.classifier.register_forward_hook(self.get_activation("pooling"))

    def close(self):
        self.model = None
        self._logger.info(f"Model Closed successfully")

    def postprocess(self, output):
        return output

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def get_features(self, img):
        with torch.no_grad():
            output = torch.nn.functional.softmax(self.model(img), dim=1)
        # utils.pick_n_best(predictions=output, n=5)
        return self.activation["pooling"], output

    def predict(self, image: np.asarray):
        # rgb to bgr
        image = image[..., ::-1]
        return self.model.get(image)

    @log_time
    def process(self, ctx: Dict[str, Any]):
        """
        ctx signature:
            image (np.asarray): Image
        """
        image: np.asarray = ctx.get("image")
        faces = self.predict(image)
        face_crops = []
        encodings = []
        detections = []
        for face in faces:
            x1, y1, x2, y2 = list(map(lambda p: max(0, int(p)), face["bbox"]))
            if (x2 - x1) * (y2 - y1) > 0:
                face_crops.append(image[y1:y2, x1:x2])
                encodings.append(face["embedding"])
                detections.append({"face": [x1, y1, x2, y2], "keypoints": face["kps"]})

        return {
            "encodings": encodings,
            "face_crops": face_crops,
            "detections": detections,
            **ctx,
        }

    @log_time
    def process_batch(self, ctx: Dict[str, Any]) -> Any:
        """
        ctx signature:
            image (List[np.asarray]): Image
        """
        images = ctx.get("image")
        face_crops = []
        encodings = []
        detections = []

        for image in images:
            e = []
            fc = []
            det = []
            faces = self.predict(image)
            for face in faces:
                x1, y1, x2, y2 = list(map(lambda p: max(0, int(p)), face["bbox"]))
                if (x2 - x1) * (y2 - y1) > 0:
                    face_crops.append(image[y1:y2, x1:x2])
                    encodings.append(face["embedding"])
                    detections.append(
                        {"face": [x1, y1, x2, y2], "keypoints": face["kps"]}
                    )

            face_crops.append(fc)
            encodings.append(e)
            detections.append(det)

        return {
            "encodings": encodings,
            "face_crops": face_crops,
            "detections": detections,
            **ctx,
        }
