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


class InsightFace(ModelProcessor):
    def __init__(
        self,
        root: str = None,
        name:str= "buffalo_l",
        model_source: Dict[str, str] = None,
        target_size=(160, 160),
        *args,
        **kwargs,
    ) -> None:
        # inject default model_source if model_path and model_source is None

        self.root = root or BATCHFLOW_HOME
        self.name =  name

        if model_source is not None:
            self._download_models(model_source, root=os.path.join(self.root, "models", self.name))
        super().__init__(
            model_path=os.path.join(BATCHFLOW_HOME, "models", self.name), model_source=model_source, *args, **kwargs
        )
        self.target_size = target_size
        self.model = FaceAnalysis(name=self.name, root=self.root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    @staticmethod
    def _download_models(model_source, root):
        for src in model_source:
            source = src["source"].lower()
            if source == "backblaze":
                bucket_name = src["bucket_name"]
                model_key: str = src["key"]
                filename: str = src["filename"]
                os.makedirs(root, exist_ok=True)
                output = os.path.join(root, filename)
                if not os.path.isfile(output):
                    storage = get_storage("backblaze", bucket_name=bucket_name)
                    storage.authenticate()
                    model_path = storage.download(output, id=model_key)

    def open(self):
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        self._logger.info(f"Loaded Model successfully")

    def close(self):
        self.model = None
        self._logger.info(f"Model Closed successfully")

    def predict(self, image: np.asarray) -> np.asarray:
        # convert np array to tf Tensor
        image_tensor: tf.Tensor = tf.convert_to_tensor(image)
        if len(image_tensor.shape) == 3:
            # add batch
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        output: np.asarray = self.model(image_tensor).numpy()
        return output

    def postprocess(self, output):
        return output

    @log_time
    def process(self, ctx: Dict[str, Any]):
        """
        ctx signature:
            image (List[np.asarray]): Image
        """
        image: np.asarray = ctx.get("image")
        faces = self.model.get(image[..., ::-1])
        face_crops = []
        encodings = []
        detections = []
        for face in faces:
            x1, y1, x2, y2 = list(map(int, face["bbox"]))
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
            face_crops (List[List[np.asarray]]): List of List of faces in RGB format
            filename (List[str]):  image filename
            image_id (List[str], optional): unique image id
            detections (List[List[Dict[str, List[Any]]]]): Listface_detection of the format
                        {"face":[x1,y1,x2,y2], "landmarks": {"right_eye":[x,y], "left_eye":[x,y], "nose":[x,y]}}
        """
        face_crops: np.asarray = ctx.get("face_crops")

        preprocessed_images = []
        indexes = []

        # make preprocessed images flat
        for i, crops in enumerate(face_crops):
            indexes.extend([i] * len(crops))
            preprocessed_images.extend(self.preprocess(crops))

        # batch the preprocessed image
        batches = []
        # batch 1
        batch = []
        for preprocessed_image in preprocessed_images:
            batch.append(preprocessed_image)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        # append last batch
        if len(batch) > 0:
            batches.append(batch)

        # process batch
        encodings_flat = []
        for batch in batches:
            # concat images in batch
            input_image: np.asarray = np.stack(batch, axis=0)
            output = [np.expand_dims(o, axis=0) for o in self.predict(input_image)]
            encodings_flat.extend(output)

        encodings = []
        # l = len(np.unique(indexes)) +1
        l = len(face_crops)
        # create empty lists
        for i in range(l):
            encodings.append([])
        for idx, enc in zip(indexes, encodings_flat):
            encodings[idx].append(enc)
        return {"encodings": encodings, **ctx}

    def preprocess(
        self,
        face_crops: List[np.asarray],
        normalization: str = "base",
    ) -> List[np.asarray]:
        """
        Preprocess input face image
        Resize the face
        Normalize the face

        Args:
            face_crops (List[np.asarray]): List of face crop
            normalization (str, optional): [image noramlization options (base, arcface)]. Defaults to "base".

        Returns:
            List[np.asarray]: [preprocessed face crops]
        """
        preprocessed_face_crops = []
        for face_crop in face_crops:
            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                factor_0 = self.target_size[0] / face_crop.shape[0]
                factor_1 = self.target_size[1] / face_crop.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (
                    int(face_crop.shape[1] * factor),
                    int(face_crop.shape[0] * factor),
                )
                face_crop = cv2.resize(face_crop, dsize)

                # Then pad the other side to the target size by adding black pixels
                diff_0 = self.target_size[0] - face_crop.shape[0]
                diff_1 = self.target_size[1] - face_crop.shape[1]
                if self.grayscale == False:
                    # Put the base face_crop in the middle of the padded face_crop
                    face_crop = np.pad(
                        face_crop,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )
                else:
                    face_crop = np.pad(
                        face_crop,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                        ),
                        "constant",
                    )

            # ------------------------------------------

            # double check: if target face_crop is not still the same size with target.
            if face_crop.shape[0:2] != self.target_size:
                face_crop = cv2.resize(face_crop, self.target_size)

            if normalization == "base":
                # normalize [0,1]
                face_crop = face_crop.astype("float")
                face_crop /= 255
            elif normalization == "arcface":
                # take from deepface normalization
                face_crop -= 127.5
                face_crop /= 128
            preprocessed_face_crops.append(face_crop)

        return preprocessed_face_crops
