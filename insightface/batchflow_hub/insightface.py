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
import onnxruntime as ort


class InsightFace(ModelProcessor):
    def __init__(
        self,
        root: str = None,
        name: str = "buffalo_l",
        model_source: Dict[str, str] = None,
        target_size=(160, 160),
        *args,
        **kwargs,
    ) -> None:
        # inject default model_source if model_path and model_source is None

        self.root = root or BATCHFLOW_HOME
        self.name = name

        if model_source is not None:
            self._download_models(
                model_source, root=os.path.join(self.root, "models", self.name)
            )
        super().__init__(
            model_path=os.path.join(BATCHFLOW_HOME, "models", self.name),
            model_source=model_source,
            *args,
            **kwargs,
        )
        self.target_size = target_size
        if ort.get_device() == "GPU":
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._logger.info(f"Using providers : {providers}")
        self.model = FaceAnalysis(
            name=self.name,
            root=self.root,
            providers=providers,
        )

    @staticmethod
    def _download_models(model_source, root):
        for src in model_source:
            source = src["source"].lower()
            if source == "backblaze":
                bucket_name = src["bucket_name"]
                key: str = src["key"]
                id: str = src.get("id", None)
                filename: str = src["filename"]
                os.makedirs(root, exist_ok=True)
                output = os.path.join(root, filename)
                if not os.path.isfile(output):
                    BUCKET_NAME = os.getenv("BUCKET_NAME")
                    B2_APPLICATION_KEY_ID = os.getenv("B2_APPLICATION_KEY_ID")
                    B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
                    B2_ENDPOINT_URL = os.getenv("B2_ENDPOINT_URL")
                    storage = get_storage(
                        "s3",
                        bucket_name=BUCKET_NAME,
                        endpoint_url=B2_ENDPOINT_URL,
                        aws_access_key_id=B2_APPLICATION_KEY_ID,
                        aws_secret_access_key=B2_APPLICATION_KEY,
                        force=True,
                    )
                    # storage = get_storage("backblaze", bucket_name=bucket_name)
                    # storage.authenticate()
                    model_path = storage.download(
                        output=output,
                        key=key,
                    )

    def open(self):
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        self._logger.info(f"Loaded Model successfully")

    def close(self):
        self.model = None
        self._logger.info(f"Model Closed successfully")

    def postprocess(self, output):
        return output

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
        bbox_data = []

        for face in faces:
            x1, y1, x2, y2 = list(map(lambda p: max(0, int(p)), face["bbox"]))

            if (x2 - x1) * (y2 - y1) > 0:
                h, w = image.shape[:2]
                # area_ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
                kps = face["kps"]
                area_ratio = 0
                try:
                    area_ratio = (
                        ((kps[1][0] - kps[0][0]) ** 2 + (kps[1][1] - kps[0][1]) ** 2)
                        ** (1 / 2)
                    ) / (h + w)
                except:
                    pass
                bbox_data.append(
                    {
                        "x1": x1 / w,
                        "x2": x2 / w,
                        "y1": y1 / h,
                        "y2": y2 / h,
                        "area_ratio": area_ratio,
                    }
                )
                face_crops.append(image[y1:y2, x1:x2])
                encodings.append(face["embedding"])
                detections.append({"face": [x1, y1, x2, y2], "keypoints": face["kps"]})

        return {
            "encodings": encodings,
            "face_crops": face_crops,
            "detections": detections,
            "bbox_data": bbox_data,
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
