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
        session_opt: Optional[List[str]] = None,
        pipeline_opt: Optional[List[str]] = None,
        max_faces: int = 50,
        detect_mode: Optional[str] = None,
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
        self.detect_mode = self._convert_opt_to_int(
            detect_mode or "HF_DETECT_MODE_ALWAYS_DETECT"
        )
        self.confidence_threshold = confidence_threshold
        default_opt_names = [
            "HF_ENABLE_QUALITY",
            "HF_ENABLE_FACE_ATTRIBUTE",
            "HF_ENABLE_FACE_EMOTION",
        ]
        self.session_opt = self._convert_opt_list_to_int(session_opt or default_opt_names)
        self.pipeline_opt = self._convert_opt_list_to_int(pipeline_opt or default_opt_names)
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

    def _convert_opt_to_int(self, opt_name: str) -> int:
        if isf is None:
            return 0
        return getattr(isf, opt_name, 0)

    def _convert_opt_list_to_int(self, opt_names: List[str]) -> int:
        if isf is None:
            return 0
        opt = 0
        for name in opt_names:
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
            encodings,
            detections,
        ) = self._run_session_on_image(image)

        return {
            "encodings": encodings,
            "detections": detections,
            **ctx,
        }

    @log_time
    def process_batch(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        images: Sequence[np.ndarray] = ctx.get("image") or []
        batched_encodings: List[List[Optional[np.ndarray]]] = []
        batched_detections: List[List[Dict[str, Any]]] = []

        for image in images:
            (
                encodings,
                detections,
            ) = self._run_session_on_image(image)
            batched_encodings.append(encodings)
            batched_detections.append(detections)
        return {
            "encodings": batched_encodings,
            "detections": batched_detections,
            **ctx,
        }

    def _run_session_on_image(
        self, image: Optional[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Dict[str, Any]], List[Dict[str, float]]]:
        if image is None:
            raise ValueError("Missing 'image' in context.")

        faces, face_extensions, image_bgr = self.predict(image)
        encodings: List[Optional[np.ndarray]] = []
        detections: List[Dict[str, Any]] = []

        height, width = image.shape[:2]

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = self._clip_box(face.location, width, height)
            if (x2 - x1) * (y2 - y1) <= 0:
                continue

            # crop = image[y1:y2, x1:x2]
            # face_crops.append(crop)
            if self.session_opt & isf.HF_ENABLE_FACE_RECOGNITION:
                encoding = self._extract_embedding(image, face)
                encodings.append(encoding)
            
            # blur_score = self._calculate_blur(crop)
            bbox_entry = {
                "x1": x1 / max(width, 1),
                "x2": x2 / max(width, 1),
                "y1": y1 / max(height, 1),
                "y2": y2 / max(height, 1),
                "area_ratio": ((x2 - x1) * (y2 - y1)) / max(width * height, 1),
                "height": height,
                "width": width,
            }
            detection_entry = self._build_detection_entry(
                face,
                face_extensions,
                bbox_entry,
                idx
            )
            detections.append(detection_entry)
        return encodings, detections

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
        self, image: np.ndarray, face: Any
    ) -> Optional[np.ndarray]:
        embeddings = self._session.face_feature_extract(image, face)
        embeddings = embeddings.reshape(-1)
        return np.asarray(embeddings, dtype=np.float32)

    def _build_detection_entry(
        self,
        face: Any,
        extensions: List[Any],
        bbox: List[int],
        index: int,
    ) -> Dict[str, Any]:
        extension = extensions[index] if index < len(extensions) else None
        conf = getattr(face, "detection_confidence", None)
        roll = getattr(face, "roll", None)
        yaw = getattr(face, "yaw", None)
        pitch = getattr(face, "pitch", None)
        attributes = self._extract_attributes(extension)
        detection_entry: Dict[str, Any] = {
            "bbox": bbox,
            "confidence": conf,
            "pose": {"pitch": pitch, "yaw": yaw, "roll": roll},
            **attributes,
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
        attributes = {}
        if self.session_opt & isf.HF_ENABLE_QUALITY:
            quality = getattr(extension, "quality_confidence", None)
            attributes["quality"] = quality
        if self.session_opt & isf.HF_ENABLE_FACE_ATTRIBUTE:
            race_idx = getattr(extension, "race", 0)
            gender_idx = getattr(extension, "gender", 0)
            age_idx = getattr(extension, "age_bracket", 0)
            emotion_idx = getattr(extension, "emotion", 0)
            attributes["race"] = self._build_categorical_attribute(race_idx, self.race_tags)
            attributes["gender"] = self._build_categorical_attribute(gender_idx, self.gender_tags)
            attributes["age"] = self._build_categorical_attribute(age_idx, self.age_bracket_tags)
        if self.session_opt & isf.HF_ENABLE_FACE_EMOTION:
            emotion_idx = getattr(extension, "emotion", 0)
            attributes["emotion"] = self._build_categorical_attribute(emotion_idx, self.emotion_tags)
        if self.session_opt & isf.HF_ENABLE_LIVENESS:
            liveness_score = getattr(extension, "rgb_liveness_confidence", None)
            attributes["liveness_score"] = liveness_score
        if self.session_opt & isf.HF_ENABLE_INTERACTION:
            left_eye_open_confidence = getattr(extension, "left_eye_status_confidence", None)
            right_eye_status_confidence = getattr(extension, "right_eye_status_confidence", None)
            attributes["left_eye_open_confidence"] = left_eye_open_confidence
            attributes["right_eye_status_confidence"] = right_eye_status_confidence
        if self.session_opt & isf.HF_ENABLE_LIVENESS:
            liveness_score = getattr(extension, "rgb_liveness_confidence", None)
            attributes["liveness_score"] = liveness_score
        
        return attributes

    def _calculate_blur(self, crop: Optional[np.ndarray]) -> Optional[float]:
        if crop is None or crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
