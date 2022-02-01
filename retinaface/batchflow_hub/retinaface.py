from sys import float_repr_style
from batchflow.processors.core import ModelProcessor
from .retinaface_lib.model import build_model
import keras
from typing import Any, Dict, List
import numpy as np
import cv2
import tensorflow as tf
from .retinaface_lib.common import preprocess, postprocess
from batchflow.decorators import log_time


class RetinaFace(ModelProcessor):
    def __init__(
        self,
        model_path: str = None,
        model_source: Dict[str, str] = None,
        threshold: float_repr_style = 0.9,
        alignface=True,
        *args,
        **kwargs,
    ) -> None:
        """
        Initalise RetinaFace Detector
        model and code taken from https://github.com/serengil/retinaface


        Args:
            model_path (str, optional): [path to model weights file]. Defaults to None.
            model_source (Dict[str, str], optional): [cloud storage ref to download the model to BATCHFLOW_HOME dir,
            refer available model_source option <>].
            Defaults to {
                source: gdrive,
                id: 1ALrrsY54fAdnmB4Zgp7yXw1-xjekiiMO,
                filename: retinaface.h5
            }
            threshold (float): face detection confidence threshold, Defaults to 0.9
        """

        if model_path is None and model_source is None:
            model_source = {
                "source": "gdrive",
                "id": "1ALrrsY54fAdnmB4Zgp7yXw1-xjekiiMO",
                "filename": "retinaface.h5",
            }
        super().__init__(model_path, model_source, *args, **kwargs)
        self.nms_threshold = 0.4
        self.decay4 = 0.5
        self._feat_stride_fpn = [32, 16, 8]
        self._anchors_fpn = {
            "stride32": np.array(
                [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]],
                dtype=np.float32,
            ),
            "stride16": np.array(
                [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]],
                dtype=np.float32,
            ),
            "stride8": np.array(
                [[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32
            ),
        }
        self._num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}
        self.allow_upscaling = True
        self.threshold = threshold
        self.alignface = alignface

    def open(self):
        self.model: keras.Model = build_model()
        self.model.load_weights(self.model_path)
        self._logger.info(f"Loaded Model successfully")

    def close(self):
        self.model = None

    def predict(self, image: np.asarray) -> tf.Tensor:
        """Detect faces in the image

        Args:
            image (np.asarray): input image

        Returns:
            tf.Tensor: Model output
        """
        image_tensor: tf.Tensor = tf.convert_to_tensor(image)
        if len(image_tensor.shape) == 3:
            # add batch
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        return self.model(image_tensor)

    @log_time
    def process(self, ctx: Dict[str, Any]) -> Any:
        """Detects face in the image

        Args:
            ctx (Dict[str, Any]): ctx object passed between nodes

        Returns:
            Any: ctx after adding output of this node
        """
        image: np.asarray = ctx.get("image")
        pre_image, im_info, im_scale, im_pad = self.preprocess(image, pad=False)
        output = self.predict(pre_image)
        detections, face_crops = self.postprocess(image, output, im_info, im_scale, im_pad)
        return {"detections": detections, "face_crops": face_crops, **ctx}

    @log_time
    def process_batch(self, ctx: Dict[str, Any]) -> Any:
        images: List[np.asarray] = ctx.get("image")
        batches = []
        # first batch
        batch = []
        for image in images:
            batch.append(image)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        # last batch
        if len(batch) > 0:
            batches.append(batch)

        # process batch

        detections = []
        face_crops = []
        for batch in batches:
            pre_image_batch = []
            meta = []
            for image in batch:
                pre_image, im_info, im_scale, im_pad = self.preprocess(
                    image, pad=True
                )
                pre_image_batch.append(pre_image)
                meta.append([im_info,im_scale, im_pad])


            pre_image_batch = np.concatenate(pre_image_batch, axis=0)
            batch_output = self.predict(pre_image_batch)
            for i, (im_info, im_scale, im_pad) in enumerate(meta):
                output = [tf.expand_dims(elt[i], axis=0) for elt in batch_output]
                dets, faces = self.postprocess(images[i], output, im_info, im_scale, im_pad)
                detections.append(dets)
                face_crops.append(faces)
        return {"detections": detections, "face_crops": face_crops, **ctx}

    def preprocess(self, image: np.asarray, pad:bool):
        """
        preprocess image before passing to model
        Args:
            image (np.asarray): [description]
            pad (bool): pad the image 

        Returns:
            [type]: [description]
        """
        return preprocess.preprocess_image(image, self.allow_upscaling, pad=pad)

    def postprocess(self, image: np.asarray, model_output: Any, im_info, im_scale, im_pad=None):
        """
        Post Process detections

        Args:
            image (np.asarray): orignal image
            model_output (Any): raw model output
            im_info ([type]): image info for postprocessing generated by preprocessing
            im_scale ([type]): image scale to rescale detections, generated by preprocessing

        Returns:
            List[Any]: detections and face_crops
        """

        # image to uint8
        image = image.astype("uint8")

        net_out = [elt.numpy() for elt in model_output]
        sym_idx = 0
        proposals_list = []
        scores_list = []
        landmarks_list = []

        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = "stride%s" % s
            scores = net_out[sym_idx]
            scores = scores[:, :, :, self._num_anchors["stride%s" % s] :]

            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

            A = self._num_anchors["stride%s" % s]
            K = height * width
            anchors_fpn = self._anchors_fpn["stride%s" % s]
            anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.reshape((-1, 1))

            bbox_stds = [1.0, 1.0, 1.0, 1.0]
            bbox_deltas = bbox_deltas
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
            proposals = postprocess.bbox_pred(anchors, bbox_deltas)

            proposals = postprocess.clip_boxes(proposals, im_info[:2])

            if s == 4 and self.decay4 < 1.0:
                scores *= self.decay4

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= self.threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            
            if im_pad is not None:
                proposals[:, 0] -= im_pad[2] # x1 - pad-left
                proposals[:, 2] -= im_pad[2] # x2 - pad-left
                proposals[:, 1] -= im_pad[0]  # y1 - pad-top
                proposals[:, 3] -= im_pad[0]  # y1 - pad-top
            proposals[:, 0:4] /= im_scale

            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_deltas = net_out[sym_idx + 2]
            landmark_pred_len = landmark_deltas.shape[3] // A
            landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
            landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            
            if im_pad is not None:
                landmarks[:, :, 0] -= im_pad[2] # x - pad-left
                landmarks[:, :, 1] -= im_pad[0] # y - pad-top
            landmarks[:, :, 0:2] /= im_scale

            landmarks_list.append(landmarks)
            sym_idx += 3

        proposals = np.vstack(proposals_list)
        if proposals.shape[0] == 0:
            detections = []
            face_crops = []
            if self.no_detection == "face":
                detections.append({"face":[0,0,image.shape[1],image.shape[0]]})
                face_crops.append(image)
            return detections, face_crops
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

        # nms = cpu_nms_wrapper(nms_threshold)
        # keep = nms(pre_det)
        # TODO: implement cuda nms
        keep = postprocess.cpu_nms(pre_det, self.nms_threshold)

        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        landmarks = landmarks[keep]

        detections = []
        face_crops = []
        for idx, face in enumerate(det):
            det = {}
            # label = 'face_'+str(idx+1)
            # resp[label]["score"] = face[4]

            det["face"] = list(face[0:4].astype(int)) + [face[4]]

            det["landmarks"] = {}
            det["landmarks"]["right_eye"] = list(landmarks[idx][0])
            det["landmarks"]["left_eye"] = list(landmarks[idx][1])
            det["landmarks"]["nose"] = list(landmarks[idx][2])
            det["landmarks"]["mouth_right"] = list(landmarks[idx][3])
            det["landmarks"]["mouth_left"] = list(landmarks[idx][4])

            

            x1, y1, x2, y2 = det["face"][:4]

            # add 5% margin
            m_x, m_y = int((0.08 * (x2-x1))//2) , int((0.1 * (y2-y1))//2)
            x1,y1,x2,y2 = x1-m_x, y1-m_y, x2 + m_x, y2+m_y 

            x1,y1 = max(0,x1), max(0,y1)
            if (x2-x1)*(y2-y1)!=0:
                face_crop = image.copy()[y1:y2, x1:x2]
                if self.alignface:
                    _landmarks = det.get("landmarks", None)
                    if _landmarks is None:
                        raise (
                            "Pass landmarks in detection or pass alignment=False in init"
                        )
                    face_crop = postprocess.alignment_procedure(
                        face_crop,
                        _landmarks["right_eye"],
                        _landmarks["left_eye"],
                        _landmarks["nose"],
                    )

                detections.append(det)
                face_crops.append(face_crop)
        return detections, face_crops
