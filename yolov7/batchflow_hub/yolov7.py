from ._yolov7.interface import get_model
from ._yolov7.utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)
from ._yolov7.utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from ._yolov7.utils.plots import plot_one_box
from typing import Dict, Tuple, Union
from batchflow.processors.core import ModelProcessor
from typing import List, Any
import numpy as np
import cv2
from batchflow.decorators import log_time
from batchflow.storage import get_storage
import os
from batchflow.constants import BATCHFLOW_HOME
import torch
from torchvision import transforms
from PIL import Image
from argparse import Namespace

import random


class YOLOv7(ModelProcessor):
    def __init__(
        self,
        root: str = None,
        name: str = "yolov7.pt",
        model_source: Dict[str, str] = None,
        model_path=None,
        target_size=(160, 160),
        *args,
        **kwargs,
    ) -> None:
        # inject default model_source if model_path and model_source is None

        self.img_size = 640
        self.stride = 32

        self.target_size = target_size
        self.opt = Namespace(
            half=False,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=0,
            agnostic_nms=False,
            augment=False,
        )
        self.model = None
        self.name = name
        self.device = select_device(0 if torch.cuda.is_available() else "cpu")
        super().__init__(
            model_path=model_path,
            model_source=model_source,
            *args,
            **kwargs,
        )

    def open(self):
        self.model = get_model(self.model_path, self.device)
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def close(self):
        self.model = None
        self._logger.info(f"Model Closed successfully")

    @staticmethod
    def letterbox(
        img,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return img, ratio, (dw, dh)

    def postprocess(self, pred, im0, img):
        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.opt.conf_thres,
            self.opt.iou_thres,
            classes=self.opt.classes,
            agnostic=self.opt.agnostic_nms,
        )
        output = []
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            save_txt = False
            save_img = False
            view_img = False

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                output.append(det.detach().cpu().numpy())

                # Print results
                # s = ""
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             if save_txt:  # Write to file
        #                 xywh = (
        #                     (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
        #                     .view(-1)
        #                     .tolist()
        #                 )  # normalized xywh
        #                 line = (
        #                     (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
        #                 )  # label format
        #                 with open(txt_path + ".txt", "a") as f:
        #                     f.write(("%g " * len(line)).rstrip() % line + "\n")

        #             if save_img or view_img:  # Add bbox to image
        #                 label = f"{self.names[int(cls)]} {conf:.2f}"
        #                 plot_one_box(
        #                     xyxy,
        #                     im0,
        #                     label=label,
        #                     color=self.colors[int(cls)],
        #                     line_thickness=1,
        #                 )
        # if view_img:
        #     cv2.imwrite("out.jpg", im0)
        return output

    def preprocess(self, img):
        # Padded resize
        img = self.letterbox(img, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.opt.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, image: np.asarray):
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(image)[0]
        return pred

    @log_time
    def process(self, ctx: Dict[str, Any]):
        """
        ctx signature:
            image (np.asarray): Image
        """
        image: np.asarray = ctx.get("image")
        pre_process_image = self.preprocess(image)
        pred = self.predict(pre_process_image)
        output = self.postprocess(pred, image, pre_process_image)

        return {
            "human_detection": output,
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
