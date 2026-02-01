"""
Module: Yolo Ingredient Service
=============================

Dịch vụ nhận diện nguyên liệu sử dụng mô hình YOLO (ONNX Runtime).

Quy trình xử lý (Pipeline):
1. Preprocess: Resize và chuẩn hóa ảnh đầu vào.
2. Inference: Chạy model ONNX để detect object.
3. Postprocess: Lọc ngưỡng confidence và NMS (nếu cần).
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

    def _load_class_names(self, data_yaml_path: str) -> list[str]:

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

    def detect_objects(self, image_bytes: bytes):
        """
        Thực hiện inference YOLO để lấy raw predictions.
        Args:
            image_bytes (bytes): Dữ liệu ảnh.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image bytes")
        img = self._preprocess(image)

        outputs = self.session.run(None, {self.input_name: img})
        predictions = outputs[0]

        if predictions.ndim == 3:
            predictions = predictions[0]

        predictions = predictions.transpose()
        
        boxes = []
        confidences = []
        class_ids = []

        for pred in predictions:
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence >= self.conf_threshold:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                
                x = cx - w / 2
                y = cy - h / 2
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.45)

        detections = []
        if len(indices) > 0:
            original_h, original_w = image.shape[:2]
            scale_x = original_w / self.input_size
            scale_y = original_h / self.input_size

            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                
                detections.append({
                   "raw_label": self.class_names[class_ids[i]],
                   "confidence": confidences[i]
                })

        return detections

    def normalize_ingredients(self, detections):
        """
        Chuyển đổi raw label (EN) sang tên hiển thị (VN) thông qua từ điển cấu hình.
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
        img = img[:, :, ::-1] 
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img


if __name__ == "__main__":
    yolo_service = YoloIngredientService()

    test_image_path = "test_images/salad.jpg"
    with open(test_image_path, "rb") as f:
        img_bytes = f.read()
    ingredients = yolo_service.detect_ingredients(img_bytes)
    print("Detected ingredients:", ingredients)