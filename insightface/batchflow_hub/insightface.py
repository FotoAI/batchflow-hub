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
    # Callers can check this before sending {"image": bgr_array, "is_bgr": True}.
    # An older release ignores the flag and would read the array as RGB.
    supports_bgr_input = True

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
            model_path=os.path.join(self.root, "models", self.name),
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

    def predict(self, image: np.asarray, is_bgr: bool = False):
        # insightface wants BGR. `image[..., ::-1]` is a view, but cv2.resize on that
        # reversed view makes a full extra frame, so callers that already hold BGR
        # (is_bgr=True) skip it. The default stays RGB input.
        if not is_bgr:
            image = image[..., ::-1]
        return self.model.get(image)

    # orientation helpers
    @staticmethod
    def _is_upright(kps: np.ndarray, tol: float = 2.0) -> bool:
        """True if the 5 InsightFace landmarks imply an upright face.

        Order is ``[left_eye, right_eye, nose, left_mouth, right_mouth]``.
        Upright requires both eyes above the nose, the nose above both mouth
        corners (image y grows downwards), and left_eye.x < right_eye.x.
        """
        le, re, nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
        eyes_y = max(float(le[1]), float(re[1]))
        mouth_y = min(float(lm[1]), float(rm[1]))
        return bool(
            eyes_y < float(nose[1]) - tol
            and float(nose[1]) < mouth_y - tol
            and float(le[0]) < float(re[0]) - tol
        )

    @staticmethod
    def _rotate_points(points: np.ndarray, k: int, src_shape) -> np.ndarray:
        """Map points from a frame of shape (H, W) into np.rot90(frame, k)."""
        H, W = src_shape[0], src_shape[1]
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        if k % 4 == 0:
            out = np.stack([x, y], axis=1)
        elif k % 4 == 1:
            out = np.stack([y, (W - 1) - x], axis=1)
        elif k % 4 == 2:
            out = np.stack([(W - 1) - x, (H - 1) - y], axis=1)
        else:
            out = np.stack([(H - 1) - y, x], axis=1)
        return out.reshape(np.asarray(points).shape)

    @classmethod
    def _unrotate_points(cls, points: np.ndarray, k: int, rot_shape) -> np.ndarray:
        """Inverse of _rotate_points: rotated-frame coords -> original coords.
        rot_shape is the ROTATED frame's (H, W)."""
        return cls._rotate_points(points, (4 - (k % 4)) % 4, rot_shape)

    @classmethod
    def _unrotate_bbox(cls, bbox, k: int, rot_shape):
        """Un-rotate a bbox by mapping its 4 corners and taking the enclosing box."""
        x1, y1, x2, y2 = bbox
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32,
        )
        back = cls._unrotate_points(corners, k, rot_shape)
        xs, ys = back[:, 0], back[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    @classmethod
    def _pick_rotation_from_kps(cls, kps: np.ndarray) -> int:
        """Return k in {0,1,2,3} such that rotating by that k makes the kps
        upright. If none fit, returns 0."""
        xs, ys = kps[:, 0], kps[:, 1]
        w = float(max(xs) - min(xs)) + 1.0
        h = float(max(ys) - min(ys)) + 1.0
        for k in (0, 1, 2, 3):
            if cls._is_upright(cls._rotate_points(kps, k, (h, w))):
                return k
        return 0

    def _sweep_rotations(self, image: np.ndarray, is_bgr: bool):
        """Return (faces, rotated_frame, rot_k).

        rot_k is the np.rot90 k that was applied (0..3). Degrees are
        ``90 * rot_k``. Coordinates in ``faces`` are in ``rotated_frame``.
        The caller is responsible for mapping them back to the original
        frame if needed.

        Tries each np.rot90 rotation (k=0..3) at the model's default threshold
        and returns the first rotation whose first detected face has upright
        keypoints. If no rotation produces upright keypoints but at least one
        produces any detection, returns the first such detection so face_align
        can still normalise minor roll. If no rotation produces any detection
        at all, returns ``([], image, 0)``.
        """
        fallback = None
        for k in (0, 1, 2, 3):
            rotated = image if k == 0 else np.ascontiguousarray(np.rot90(image, k))
            faces = self.predict(rotated, is_bgr=is_bgr)
            if not faces:
                continue
            if self._is_upright(faces[0]["kps"]):
                return faces, rotated, k
            if fallback is None:
                fallback = (faces, rotated, k)
        if fallback is not None:
            return fallback
        return [], image, 0

    def _encode_each_face_upright(
        self,
        frame: np.ndarray,
        faces: List[Any],
        is_bgr: bool,
    ) -> List[Any]:
        """For each face whose kps are not upright, pad+crop its region out
        of ``frame``, rotate the crop so the kps become upright, re-run
        detection on the rotated crop to get fresh (upright) kps, and
        REPLACE the face's embedding with the one computed on that upright
        crop. The returned face dicts keep their original ``bbox`` and
        ``kps`` (in ``frame`` coords) so caller-side un-rotation stays
        correct; only ``embedding`` is updated.
        """
        H, W = frame.shape[:2]
        out = []
        for face in faces:
            if self._is_upright(face["kps"]):
                out.append(face)
                continue
            k_f = self._pick_rotation_from_kps(face["kps"])
            if k_f == 0:
                out.append(face)
                continue
            x1, y1, x2, y2 = [float(v) for v in face["bbox"]]
            bw, bh = x2 - x1, y2 - y1
            pad = 0.5 * max(bw, bh)
            px1 = max(0, int(x1 - pad))
            py1 = max(0, int(y1 - pad))
            px2 = min(W, int(x2 + pad))
            py2 = min(H, int(y2 + pad))
            padded = frame[py1:py2, px1:px2]
            if padded.size == 0:
                out.append(face)
                continue
            rotated_crop = np.ascontiguousarray(np.rot90(padded, k_f))
            new_faces = self.predict(rotated_crop, is_bgr=is_bgr)
            if not new_faces:
                out.append(face)
                continue
            # Pick the largest newly-detected face in the rotated crop.
            best = max(
                new_faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0])
                              * (f["bbox"][3] - f["bbox"][1]),
            )
            # dict() shallow-copies the stock insightface Face dict; keep
            # bbox and kps as-is so downstream coords remain valid.
            updated = dict(face)
            updated["embedding"] = best["embedding"]
            out.append(updated)
        return out

    @log_time
    def process(self, ctx: Dict[str, Any]):
        """
        ctx signature:
            image (np.asarray): Image, RGB unless is_bgr is set
            is_bgr (bool, optional): the image is already BGR. Default False.
                Face crops are views of the image, so they keep its channel order.
            rotation_sweep (bool, optional): when True, try the four np.rot90
                rotations at the detector's default threshold and keep the one
                whose first face has upright keypoints (eyes above nose above
                mouth, left eye to the left of the right eye). This gives the
                detector an upright face so SCRFD returns clean landmarks and
                face_align.norm_crop then has a trivial 2D transform to apply.
                Falls back to the first non-empty rotation if none is upright.
                The returned ``image`` is always the ORIGINAL input and
                bboxes / keypoints / face crops are mapped back to its
                coordinates; the chosen rotation is only used internally.
                ``applied_rotation`` records the degrees applied internally.
                Default False (byte-for-byte identical to the stock path).
            per_face_rotation (bool, optional): only meaningful together with
                rotation_sweep. When True, every detected face whose keypoints
                are still not upright after the whole-image sweep gets its
                padded crop rotated individually and re-encoded, so a group
                photo with mixed-orientation faces can produce an upright
                embedding per face. Default False.
        """
        image: np.asarray = ctx.get("image")
        is_bgr = bool(ctx.get("is_bgr", False))
        rotation_sweep = bool(ctx.get("rotation_sweep", False))
        per_face_rotation = bool(ctx.get("per_face_rotation", False))
        if rotation_sweep:
            faces, rotated_frame, rot_k = self._sweep_rotations(image, is_bgr)
            if per_face_rotation and faces:
                faces = self._encode_each_face_upright(
                    rotated_frame, faces, is_bgr,
                )
        else:
            faces = self.predict(image, is_bgr=is_bgr)
            rotated_frame = image
            rot_k = 0

        # Output is always expressed against the ORIGINAL image: callers get
        # the input they handed us (unchanged) and bbox/kps that locate the
        # face in that original frame. The rotation is only applied inside
        # the pipeline for detection + encoding quality.
        rot_shape = rotated_frame.shape[:2]
        orig_h, orig_w = image.shape[:2]
        applied_rotation = 90 * rot_k

        face_crops = []
        encodings = []
        detections = []
        bbox_data = []

        for face in faces:
            if rot_k:
                bx1, by1, bx2, by2 = self._unrotate_bbox(
                    face["bbox"], rot_k, rot_shape,
                )
                kps_out = self._unrotate_points(face["kps"], rot_k, rot_shape)
            else:
                bx1, by1, bx2, by2 = face["bbox"]
                kps_out = face["kps"]

            x1 = max(0, int(bx1))
            y1 = max(0, int(by1))
            x2 = max(0, int(bx2))
            y2 = max(0, int(by2))

            if (x2 - x1) * (y2 - y1) > 0:
                # area_ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
                area_ratio = 0
                try:
                    kps_arr = np.asarray(kps_out, dtype=np.float32)
                    area_ratio = (
                        ((kps_arr[1][0] - kps_arr[0][0]) ** 2
                         + (kps_arr[1][1] - kps_arr[0][1]) ** 2) ** 0.5
                    ) / (orig_h + orig_w)
                except Exception:
                    pass
                bbox_data.append(
                    {
                        "x1": x1 / orig_w,
                        "x2": x2 / orig_w,
                        "y1": y1 / orig_h,
                        "y2": y2 / orig_h,
                        "area_ratio": area_ratio,
                    }
                )
                face_crops.append(image[y1:y2, x1:x2])
                encodings.append(face["embedding"])
                detections.append({"face": [x1, y1, x2, y2], "keypoints": kps_out})

        return {
            **ctx,
            "encodings": encodings,
            "face_crops": face_crops,
            "detections": detections,
            "bbox_data": bbox_data,
            "image": image,
            "applied_rotation": int(applied_rotation),
        }

    @log_time
    def process_batch(self, ctx: Dict[str, Any]) -> Any:
        """
        ctx signature:
            image (List[np.asarray]): Image

        Note: batch mode does not implement ``rotation_sweep``; the flag is
        silently ignored here.
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
