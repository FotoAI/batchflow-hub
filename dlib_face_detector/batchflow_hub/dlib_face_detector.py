from batchflow.processors.core import ModelProcessor
from typing import Dict, Any, List
import dlib


class DLibFaceDetector(ModelProcessor):
    def __init__(self, model_path: str = None, model_source: Dict[str, str] = None, *args, **kwargs) -> None:

        super().__init__(model_path, model_source, *args, **kwargs)
