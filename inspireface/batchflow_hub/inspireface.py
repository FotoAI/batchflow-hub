from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from batchflow.constants import BATCHFLOW_HOME
from batchflow.decorators import log_time
from batchflow.processors.core import ModelProcessor

try:
    import inspireface as isf
except ImportError:  # pragma: no cover - runtime dependency handled at execution
    isf = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency for debug annotations
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_RACE_TAGS = {
    idx: label
    for idx, label in enumerate(
        [
            "Black",
            "Asian",
            "Latino/Hispanic",
            "Middle Eastern",
            "White",
        ]
    )
}
DEFAULT_GENDER_TAGS = {
    idx: label for idx, label in enumerate(["Female", "Male"])
}
DEFAULT_AGE_BRACKET_TAGS = {
    idx: label
    for idx, label in enumerate(
        [
            "0-2 years old",
            "3-9 years old",
            "10-19 years old",
            "20-29 years old",
            "30-39 years old",
            "40-49 years old",
            "50-59 years old",
            "60-69 years old",
            "more than 70 years old",
        ]
    )
}
DEFAULT_EMOTION_TAGS = {
    idx: label
    for idx, label in enumerate(
        [
            "Neutral",
            "Happy",
            "Sad",
            "Surprise",
            "Fear",
            "Disgust",
            "Anger",
        ]
    )
}


class InspireFace(ModelProcessor):
    def __init__(
        self,
        root: Optional[str] = None,
        name: str = "Megatron",
        model_source: Optional[Dict[str, str]] = None,
        target_size: Tuple[int, int] = (160, 160),
        session_opt: Optional[int] = None,
        pipeline_opt: Optional[int] = None,
        max_faces: int = 50,
        detect_mode: Optional[int] = None,
        detect_pixel_level: int = -1,
        confidence_threshold: float = 0.4,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.root = root or BATCHFLOW_HOME
        self.name = name
        debug_root = debug_dir or os.path.join(BATCHFLOW_HOME, "debug", "inspireface")

        super().__init__(
            model_path=os.path.join(self.root, "models", self.name),
            model_source=model_source,
            *args,
            **kwargs,
        )

        self.target_size = target_size
        self.max_faces = max_faces
        self.detect_pixel_level = detect_pixel_level
        self.detect_mode = (
            detect_mode or getattr(isf, "HF_DETECT_MODE_ALWAYS_DETECT", 0)
        )
        self.confidence_threshold = confidence_threshold
        self.session_opt = session_opt or self._compute_default_session_opt()
        self.pipeline_opt = pipeline_opt or self._compute_default_pipeline_opt()
        self.debug = debug
        self.debug_dir = Path(debug_root)
        self.debug_crops_dir = self.debug_dir / "crops"
        self.debug_annotated_dir = self.debug_dir / "annotated"
        self._session: Optional[Any] = None
        self._model_launched = False
        self.race_tags = DEFAULT_RACE_TAGS
        self.gender_tags = DEFAULT_GENDER_TAGS
        self.age_bracket_tags = DEFAULT_AGE_BRACKET_TAGS
        self.emotion_tags = DEFAULT_EMOTION_TAGS

        if self.debug:
            self._prepare_debug_dirs()

    def _prepare_debug_dirs(self) -> None:
        self.debug_crops_dir.mkdir(parents=True, exist_ok=True)
        self.debug_annotated_dir.mkdir(parents=True, exist_ok=True)

    def _compute_default_session_opt(self) -> int:
        if isf is None:
            return 0
        option_names = [
            "HF_ENABLE_QUALITY",
            "HF_ENABLE_INTERACTION",
            "HF_ENABLE_FACE_POSE",
            "HF_ENABLE_FACE_ATTRIBUTE",
            "HF_ENABLE_FACE_EMOTION",
            "HF_ENABLE_RECOGNITION",
        ]
        opt = 0
        for name in option_names:
            opt |= getattr(isf, name, 0)
        return opt

    def _compute_default_pipeline_opt(self) -> int:
        if isf is None:
            return 0
        option_names = [
            "HF_ENABLE_QUALITY",
            "HF_ENABLE_FACE_ATTRIBUTE",
            "HF_ENABLE_INTERACTION",
            "HF_ENABLE_FACE_EMOTION",
            "HF_ENABLE_RECOGNITION",
        ]
        opt = 0
        for name in option_names:
            opt |= getattr(isf, name, 0)
        return opt

    def _launch_model(self) -> None:
        if self._model_launched:
            return
        if isf is None:
            raise ImportError(
                "The inspireface package is required but not installed. "
                "Install it with `pip install inspireface`."
            )
        self._logger.info(
            "Launching InspireFace resources for model pack '%s'...", self.name
        )
        launch_status = isf.reload(self.name)
        if not launch_status:
            raise RuntimeError(
                f"Failed to launch InspireFace resources for pack '{self.name}'."
            )
        self._model_launched = True
        self._logger.info("Launch complete. Status: %s", isf.query_launch_status())

    def open(self) -> None:
        self._launch_model()
        self._session = isf.InspireFaceSession(
            param=self.session_opt,
            detect_mode=self.detect_mode,
            max_detect_num=self.max_faces,
            detect_pixel_level=self.detect_pixel_level,
        )
        if self.confidence_threshold is not None:
            self._session.set_detection_confidence_threshold(self.confidence_threshold)
        self._logger.info("Created InspireFaceSession (max_faces=%s)", self.max_faces)

    def close(self) -> None:
        self._session = None
        self._logger.info("InspireFace session closed")

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[Any], List[Any], Optional[np.ndarray]]:
        if self._session is None:
            raise RuntimeError("InspireFaceSession is not initialized. Call open().")

        if image is None:
            raise ValueError("Missing 'image' in context.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (H, W, 3).")

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = self._session.face_detection(image_bgr)
        face_extensions: List[Any] = []
        if faces:
            try:
                face_extensions = self._session.face_pipeline(
                    image_bgr, faces, self.pipeline_opt
                )
            except Exception as exc:  # pragma: no cover - passthrough for runtime errors
                self._logger.warning("face_pipeline failed: %s", exc)
        return faces, face_extensions, image_bgr

    @log_time
    def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        image: Optional[np.ndarray] = ctx.get("image")
        (
            face_crops,
            encodings,
            detections,
            bbox_data,
        ) = self._run_session_on_image(image)

        return {
            "encodings": encodings,
            "face_crops": face_crops,
            "detections": detections,
            "bbox_data": bbox_data,
            **ctx,
        }

    @log_time
    def process_batch(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        images: Sequence[np.ndarray] = ctx.get("image") or []
        filenames: Sequence[Optional[str]] = ctx.get("filename") or []
        batched_crops: List[List[np.ndarray]] = []
        batched_encodings: List[List[Optional[np.ndarray]]] = []
        batched_detections: List[List[Dict[str, Any]]] = []
        batched_bbox_data: List[List[Dict[str, float]]] = []

        for image in images:
            (
                face_crops,
                encodings,
                detections,
                bbox_data,
            ) = self._run_session_on_image(image)
            batched_crops.append(face_crops)
            batched_encodings.append(encodings)
            batched_detections.append(detections)
            batched_bbox_data.append(bbox_data)

        return {
            "encodings": batched_encodings,
            "face_crops": batched_crops,
            "detections": batched_detections,
            "bbox_data": batched_bbox_data,
            **ctx,
        }

    def _run_session_on_image(
        self, image: Optional[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Dict[str, Any]], List[Dict[str, float]]]:
        if image is None:
            raise ValueError("Missing 'image' in context.")

        faces, face_extensions, image_bgr = self.predict(image)
        face_crops: List[np.ndarray] = []
        encodings: List[Optional[np.ndarray]] = []
        detections: List[Dict[str, Any]] = []
        bbox_data: List[Dict[str, float]] = []

        height, width = image.shape[:2]

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = self._clip_box(face.location, width, height)
            if (x2 - x1) * (y2 - y1) <= 0:
                continue

            crop = image[y1:y2, x1:x2]
            face_crops.append(crop)
            embedding = self._extract_embedding(face_extensions, idx)
            encodings.append(embedding)
            blur_score = self._calculate_blur(crop)

            bbox_entry = {
                "x1": x1 / max(width, 1),
                "x2": x2 / max(width, 1),
                "y1": y1 / max(height, 1),
                "y2": y2 / max(height, 1),
                "area_ratio": ((x2 - x1) * (y2 - y1)) / max(width * height, 1),
            }
            bbox_data.append(bbox_entry)

            detection_entry = self._build_detection_entry(
                face,
                face_extensions,
                idx,
                [x1, y1, x2, y2],
                embedding,
                blur_score,
            )
            detections.append(detection_entry)
        return face_crops, encodings, detections, bbox_data

    def _clip_box(
        self, location: Sequence[int], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(int, location)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        return x1, y1, x2, y2

    def _extract_embedding(
        self, extensions: List[Any], index: int
    ) -> Optional[np.ndarray]:
        if not extensions or index >= len(extensions):
            return None
        extension = extensions[index]
        candidate_attrs = ("embedding", "feature", "face_feature")
        vector = None
        for attr in candidate_attrs:
            vector = getattr(extension, attr, None)
            if vector is not None:
                break
        if vector is None:
            return None
        arr = np.asarray(vector, dtype=np.float32)
        return arr.reshape(-1)

    def _extract_keypoints(self, face: Any) -> Dict[str, Tuple[float, float]]:
        candidate_attrs = (
            "dense_landmark",
            "landmark",
            "key_points",
            "keypoints",
            "kps",
        )
        keypoints = None
        for attr in candidate_attrs:
            keypoints = getattr(face, attr, None)
            if keypoints is not None:
                break
        if keypoints is None:
            return {}
        normalized = {}
        if isinstance(keypoints, dict):
            for key, value in keypoints.items():
                if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
                    normalized[key] = (float(value[0]), float(value[1]))
        else:
            for idx, value in enumerate(keypoints):
                if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
                    normalized[f"p{idx}"] = (float(value[0]), float(value[1]))
        return normalized

    def _build_detection_entry(
        self,
        face: Any,
        extensions: List[Any],
        index: int,
        bbox: List[int],
        embedding: Optional[np.ndarray],
        blur_score: Optional[float],
    ) -> Dict[str, Any]:
        extension = extensions[index] if index < len(extensions) else None
        conf = getattr(face, "detection_confidence", None)
        roll = getattr(face, "roll", None)
        yaw = getattr(face, "yaw", None)
        pitch = getattr(face, "pitch", None)
        quality = getattr(extension, "quality_confidence", None) if extension else None
        attributes = self._extract_attributes(extension)
        detection_entry: Dict[str, Any] = {
            "bbox": bbox,
            "keypoints": self._extract_keypoints(face),
            "confidence": conf,
            "pose": {"pitch": pitch, "yaw": yaw, "roll": roll},
            "quality": quality,
            "blur": blur_score,
            "race": attributes.get("race"),
            "gender": attributes.get("gender"),
            "age": attributes.get("age"),
            "emotion": attributes.get("emotion"),
        }
        return detection_entry

    def _build_categorical_attribute(
        self, idx: Optional[int], tags: Dict[int, str]
    ) -> Dict[str, Any]:
        if idx is None:
            return {"index": None, "label": "Unknown"}
        label = tags.get(idx, "Unknown")
        return {"index": int(idx), "label": label}

    def _extract_attributes(self, extension: Optional[Any]) -> Dict[str, Any]:
        if extension is None:
            return {}
        race_idx = getattr(extension, "race", 0)
        gender_idx = getattr(extension, "gender", 0)
        age_idx = getattr(extension, "age_bracket", 0)
        emotion_idx = getattr(extension, "emotion", 0)
        return {
            "race": self._build_categorical_attribute(race_idx, self.race_tags),
            "gender": self._build_categorical_attribute(gender_idx, self.gender_tags),
            "age": self._build_categorical_attribute(
                age_idx, self.age_bracket_tags
            ),
            "emotion": self._build_categorical_attribute(
                emotion_idx, self.emotion_tags
            ),
            "interaction": getattr(extension, "interaction", None),
        }

    def _calculate_blur(self, crop: Optional[np.ndarray]) -> Optional[float]:
        if crop is None or crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
