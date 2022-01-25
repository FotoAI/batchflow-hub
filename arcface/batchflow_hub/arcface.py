from typing import Dict, Tuple, Union
from batchflow.processors.core import ModelProcessor
from .model import build_model
import tensorflow as tf
import keras
from typing import List, Any
import numpy as np
import cv2
from batchflow.decorators import log_time
from .common import postprocess


class ArcFaceProcessor(ModelProcessor):
    def __init__(
        self,
        model_path: str = None,
        model_source: Dict[str, str] = None,
        target_size=(112, 112),
        grayscale=False,
        alignface=True,
        *args,
        **kwargs,
    ) -> None:
        # inject default model_source if model_path and model_source is None
        if model_path is None and model_source is None:
            model_source = {
                "source": "gdrive",
                "id": "1bUp9KTSvnHnsB9s3YkiJKVr14Rw0t7kA",
                "filename": "arcface_weights.h5",
            }
        super().__init__(
            model_path=model_path, model_source=model_source, *args, **kwargs
        )
        self.target_size = target_size
        self.grayscale = grayscale
        self.alignface = alignface

    def open(self):
        self.model: keras.Model = build_model()
        self.model.load_weights(self.model_path)
        self._logger.info(f"Loaded Model successfully")

    def close(self):
        self.model = None
        self._logger.info(f"Model Closed successfully")

    @log_time
    def predict(self, image: np.asarray) -> np.asarray:
        # convert np array to tf Tensor
        image_tensor: tf.Tensor = tf.convert_to_tensor(image)
        if len(image_tensor.shape) == 3:
            # add batch
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        output: np.asarray = self.model(image_tensor).numpy()
        return output

    def process(self, ctx: Dict[str, Any]):
        """
        ctx signature:
            image (np.asarray): image in RGB format
            filename (str):  image filename
            image_id (str, optional): unique image id
            detections (List[Dict[str, List[Any]]]): face_detection of the format
                        {"face":[x1,y1,x2,y2], "landmarks": {"right_eye":[x,y], "left_eye":[x,y], "nose":[x,y]}}
        """
        image: np.asarray = ctx.get("image")
        detections: List[Union[List, Tuple]] = ctx.get("detections")

        preprocess_images: List[np.asarray] = self.preprocess(image, detections)
        encodings: List[np.asarray] = []
        for preprocess_image in preprocess_images:
            output = self.predict(preprocess_image)
            encodings.append(self.postprocess(output))
        return {"encodings": encodings, **ctx}

    def process_batch(self, ctx: Dict[str, Any]) -> Any:
        """
        ctx signature:
            image (List[np.asarray]): batch of image in RGB format
            filename (List[str]):  image filename
            image_id (List[str], optional): unique image id
            detections (List[List[Dict[str, List[Any]]]]): Listface_detection of the format
                        {"face":[x1,y1,x2,y2], "landmarks": {"right_eye":[x,y], "left_eye":[x,y], "nose":[x,y]}}
        """
        images: List[np.asarray] = ctx.get("image")
        image_ids: List[List[Union[str, int]]] = ctx.get("image_id")
        detections: List[List[Union[List, Tuple]]] = ctx.get("detections")

        preprocessed_images = []
        indexes = []

        # make preprocessed images flat
        for i, (image, det) in enumerate(zip(images, detections)):
            indexes.extend([i] * len(det))
            preprocessed_images.extend(self.preprocess(image, det))

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
        if len(indexes) > 0:
            last_index = indexes[0]
            e = [encodings_flat[last_index]]
            for i in indexes[1:]:
                if last_index == i:
                    e.append([encodings_flat[i]])
                else:
                    encodings.append(e)
                    e = [encodings_flat[i]]
                last_index = i
            # add last image encoding
            encodings.append(e)
            e = []
        return {"encodings": encodings, **ctx}

    def preprocess(
        self,
        image: np.asarray,
        detections: List[Union[List, Tuple]],
        normalization: str = "base",
    ) -> List[np.asarray]:
        """
        Preporcess input image
        Crop the faces from the image
        Resize the face
        Normalize the face

        Args:
            image (np.asarray): [image]
            detections (List[Union[List, Tuple]]): [face detection of format [x1,y1,x2,y2]]
            normalization (str, optional): [image noramlization options (base, arcface)]. Defaults to "base".

        Returns:
            List[np.asarray]: [preprocessed face crops]
        """
        face_crops = []
        for det in detections:
            x1, y1, x2, y2, conf = det["face"]
            landmarks = det["landmarks"]
            face_crop = image.copy()[y1:y2, x1:x2]
            face_crop = postprocess.alignment_procedure(
                face_crop,
                landmarks["right_eye"],
                landmarks["left_eye"],
                landmarks["nose"],
            )

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
                face_crop /= 255
            elif normalization == "arcface":
                # take from deepface normalization
                face_crop -= 127.5
                face_crop /= 128
            face_crops.append(face_crop)

        return face_crops
