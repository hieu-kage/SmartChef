import os
from pathlib import Path


AI_SERVICES_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent

RAG_SERVICES_DIR = AI_SERVICES_DIR / "app" / "RAG"
PREPARE_DATA_DIR = AI_SERVICES_DIR / "app" / "RAG" / "prepareDataForRag"
PREPARE_SCRIPTS_DIR = PREPARE_DATA_DIR / "scripts"

YOLO_SERVICE_DIR = AI_SERVICES_DIR / "app" / "Vision"
YOLO_MODEL_PATH = YOLO_SERVICE_DIR / "Yolo_Model" / "best.onnx"
YOLO_DATA_YAML = YOLO_SERVICE_DIR / "Yolo_Model" / "data.yaml"

RECIPES_JSON_PATH = PREPARE_DATA_DIR / "smartchef_dataset.json"
