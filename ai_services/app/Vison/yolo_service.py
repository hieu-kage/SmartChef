"""
Module: Yolo Ingredient Service
=============================

Dịch vụ nhận diện nguyên liệu sử dụng mô hình YOLO (ONNX Runtime).

Quy trình xử lý (Pipeline):
1. Preprocess: Resize và chuẩn hóa ảnh đầu vào.
2. Inference: Chạy model ONNX để detect object.
3. Postprocess: Lọc ngưỡng confidence và NMS bằng NumPy (Đã tối ưu Vectorization).
4. Normalization: Ánh xạ nhãn (label) sang tiếng Việt chuẩn.
"""

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from ..config import YOLO_CLASS_TO_VI
from .. import config

class YoloIngredientService:
    """
    Service wrapper cho YOLO ONNX Model.
    """
    def __init__(
        self,
        model_path=config.paths.YOLO_MODEL_PATH,
        data_yaml_path=config.paths.YOLO_DATA_YAML,
        conf_threshold: float = 0.5
    ):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        self.input_size = (
            input_meta.shape[2]
            if isinstance(input_meta.shape[2], int)
            else 640
        )

        self.conf_threshold = conf_threshold
        self.class_names = self._load_class_names(data_yaml_path)

    def _load_class_names(self, data_yaml_path: str):
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data["names"]

    def detect_ingredients(self, image_bytes: bytes) -> list[str]:
        """
        Nhận diện và trả về danh sách tên nguyên liệu (Tiếng Việt).
        Args:
            image_bytes (bytes): Dữ liệu ảnh dạng raw bytes.
        """
        detections = self.detect_objects(image_bytes)
        return self.normalize_ingredients(detections)

    def detect_objects(self, image_bytes: bytes) -> list[dict]:
        """
        Thực hiện inference YOLO để lấy raw predictions (Đã tối ưu tốc độ bằng NumPy).
        Args:
            image_bytes (bytes): Dữ liệu ảnh.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image bytes")
            
        original_h, original_w = image.shape[:2]
        img = self._preprocess(image)

        outputs = self.session.run(None, {self.input_name: img})
        predictions = outputs[0]

        if predictions.ndim == 3:
            predictions = predictions[0]

        predictions = predictions.transpose()
        
        # --- TỐI ƯU HÓA: Vectorization ---
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        detections = []
        if len(boxes) == 0:
            return detections

        x = boxes[:, 0] - boxes[:, 2] / 2
        y = boxes[:, 1] - boxes[:, 3] / 2
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        boxes_xywh = np.column_stack((x, y, w, h)).tolist()
        confidences_list = confidences.tolist()

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences_list, self.conf_threshold, 0.45)

        if len(indices) > 0:
            for i in indices.flatten():
                # Lấy tên class an toàn hỗ trợ cả Dict và List từ file data.yaml
                c_id = class_ids[i]
                raw_label = self.class_names[c_id] if isinstance(self.class_names, (list, tuple)) else self.class_names.get(c_id, f"class_{c_id}")
                
                detections.append({
                    "raw_label": raw_label,
                    "confidence": float(confidences_list[i])
                })

        return detections

    def normalize_ingredients(self, detections: list[dict]) -> list[str]:
        """
        Chuyển đổi raw label (EN/Không dấu) sang tên hiển thị (VN) thông qua từ điển cấu hình.
        """
        ingredients = set()

        for d in detections:
            raw_label = d["raw_label"]
            vi_name = YOLO_CLASS_TO_VI.get(raw_label)

            if vi_name:
                ingredients.add(vi_name)

        return list(ingredients)

    def _preprocess(self, image):
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = img[:, :, ::-1] # BGR to RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img