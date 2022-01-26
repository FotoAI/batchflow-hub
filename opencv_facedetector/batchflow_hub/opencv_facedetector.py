from typing import Dict
from batchflow.core.node import ProcessorNode
from typing import Dict, Any, List
from .opencv_wrapper import build_model, alignment_procedure
import cv2
import numpy as np
from batchflow.decorators import log_time



class OpenCVFaceDetector(ProcessorNode):
    def __init__(self, alignface=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignface = alignface

    def open(self):
        d = build_model()
        self.face_detector = d["face_detector"]
        self.eye_detector = d["eye_detector"]
        self._logger.info("Loaded Face and Eye Detector")

    def close(self):
        self.face_detector = None
        self.eye_detector = None

    @log_time
    def process(self, ctx: Dict[str, Any]) -> any:
        """
        Detect Face using HaarCascade

        Args:
            ctx (Dict[str,Any]): ctx dict

        Returns:
            any: Face detections and face crops
        """
        image: np.asarray = ctx.get("image")
        detections, face_crops = self.detect_face(image)
        return {"detections": detections, "face_crops": face_crops, **ctx}


    @log_time
    def process_batch(self, ctx: Dict[str, Any]) -> any:
        """
        Detect Face using HaarCascade

        Args:
            ctx (Dict[str,Any]): ctx dict

        Returns:
            any: Face detections and face crops
        """
        images: List[np.asarray] = ctx.get("image")

        detections = []
        face_crops = []
        for image in images:
            dets, crops = self.detect_face(image)
            detections.append(dets)
            face_crops.append(crops)
        return {"detections": detections, "face_crops": face_crops, **ctx}

    def detect_face(self, image: np.asarray):
        """
        Detect face in the image

        Args:
            image (np.asarray): input image

        Returns:
            Tuple[List[List[Union[int, float]]], List[np.asarray]]: Face Bounding Box and Face Crops
        """

        resp = []

        detected_face = None
        img_region = [0, 0, image.shape[0], image.shape[1]]

        faces = []
        try:
            # faces = detector["face_detector"].detectMultiScale(image, 1.3, 5)
            faces = self.face_detector.detectMultiScale(image, 1.1, 10)
        except:
            pass

        detections = []
        face_crops = []

        if len(faces) > 0:
            for x, y, w, h in faces:
                detected_face = image[int(y) : int(y + h), int(x) : int(x + w)]
                landmarks = None
                if self.alignface:
                    detected_face, landmarks = self.align_face(self.eye_detector, detected_face)
                    # move landmark point wrt to full image
                    if landmarks is not None:
                        for k, (ix,iy) in landmarks.items():
                            landmarks[k] = [x+ix, y+iy]

                detections.append({"face":[x, y, x + w, y + h], "landmarks":landmarks})
                face_crops.append(detected_face)
                

        return detections, face_crops

    @staticmethod
    def align_face(eye_detector, img):

        detected_face_gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        )  # eye detector expects gray scale image

        # eyes = eye_detector.detectMultiScale(detected_face_gray, 1.3, 5)
        eyes = eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)

        if len(eyes) >= 2:

            # find the largest 2 eye
            """
            base_eyes = eyes[:, 2]

            items = []
            for i in range(0, len(base_eyes)):
                item = (base_eyes[i], i)
                items.append(item)

            df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)

            eyes = eyes[df.idx.values[0:2]] #eyes variable stores the largest 2 eye
            """
            # eyes = eyes[0:2]

            # -----------------------
            # decide left and right eye

            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            # -----------------------
            # find center of eyes
            left_eye = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)),
            )
            right_eye = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)),
            )
            img = alignment_procedure(img, left_eye, right_eye)
            return img, {"right_eye":right_eye, "left_eye":left_eye}

        return img, None # return img anyway
