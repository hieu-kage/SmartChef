from pathlib import Path


def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "docker-compose.yml").exists():
            return parent

    raise RuntimeError(
        "❌ Không tìm thấy PROJECT ROOT (thiếu docker-compose.yml)"
    )


# ===== ROOT =====
PROJECT_ROOT = find_project_root()


# ===== CORE DIRS =====
CONFIG_DIR = Path(__file__).parent
AI_SERVICES_DIR = PROJECT_ROOT / "ai_services"
VECTOR_DB_DIR = PROJECT_ROOT / "vector-db"


# ===== AI SERVICES =====
RAG_SERVICES_DIR = AI_SERVICES_DIR / "app" / "RAG"
PREPARE_DATA_DIR = AI_SERVICES_DIR/ "app" / "RAG" / "prepareDataForRag"
PREPARE_SCRIPTS_DIR = PREPARE_DATA_DIR / "scripts"

YOLO_SERVICE_DIR = AI_SERVICES_DIR / "app" / "Vison"
YOLO_MODEL_PATH = YOLO_SERVICE_DIR / "Yolo_Model" / "best.onnx"
YOLO_DATA_YAML = YOLO_SERVICE_DIR / "Yolo_Model" / "data.yaml"


RECIPES_JSON_PATH = PREPARE_DATA_DIR / "smartchef_dataset.json"


QDRANT_DATA_DIR = VECTOR_DB_DIR / "data"

