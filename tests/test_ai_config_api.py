"""
Flask AI config API tests.
"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.ai_config_store as store
from app import app


def use_temp_configs(temp_dir):
    config_dir = Path(temp_dir) / "configs"
    store.CONFIG_DIR = config_dir
    store.CONFIG_PATH = config_dir / "config.json"
    store.ensure_config_files()


def test_detection_settings_aliases():
    with TemporaryDirectory() as temp_dir:
        use_temp_configs(temp_dir)
        client = app.test_client()
        payload = {
            "detectionEnabled": True,
            "detectionModel": "240922",
            "enginePath": "weights/yolo11n_person.engine",
            "detectionThreshold": 0.6,
            "overlapRate": 0.2,
            "matchEnabled": True,
            "matchThreshold": 0.5,
            "matchFrequency": 5,
            "target": "all",
        }

        response = client.post("/api/detection/settings", json=payload)
        assert response.status_code == 200
        assert response.get_json()["status"] == "success"

        response = client.post("/api/detection/region", json=payload)
        assert response.status_code == 200
        assert response.get_json()["status"] == "success"

        config = store.get_detection_config()["config"]
        assert config["backend"] == "tensorrt"
        assert config["engine_path"] == "weights/yolo11n_person.engine"
        assert config["thresholds"]["conf_thres"] == 0.6
        assert config["overlap_threshold"] == 0.2
        assert config["match_frequency"] == 5


def test_detection_regions_rect_to_points_and_clear():
    with TemporaryDirectory() as temp_dir:
        use_temp_configs(temp_dir)
        client = app.test_client()
        payload = {
            "target": "high",
            "clear": False,
            "regions": [
                {
                    "id": 1,
                    "rect": {"x1": 100, "y1": 200, "x2": 400, "y2": 350},
                }
            ],
        }

        response = client.post("/api/detection/regions", json=payload)
        assert response.status_code == 200
        assert response.get_json()["status"] == "success"

        roi = store.get_roi_config()["rois"][0]
        assert roi["target"] == "high"
        assert roi["type"] == "cleaning"
        assert roi["roi_type"] == "clear_zone"
        assert roi["bbox"] == [100.0, 200.0, 300.0, 150.0]
        assert roi["points"] == [
            [100.0, 200.0],
            [400.0, 200.0],
            [400.0, 350.0],
            [100.0, 350.0],
        ]
        assert roi["polygon"] == roi["points"]

        response = client.post("/api/detection/regions", json={"target": "high", "clear": True})
        assert response.status_code == 200
        assert response.get_json()["message"] == "检测区域已清空"
        assert store.get_roi_config()["rois"] == []


def test_detection_regions_accepts_rois_key():
    with TemporaryDirectory() as temp_dir:
        use_temp_configs(temp_dir)
        client = app.test_client()
        payload = {
            "target": "low",
            "clear": False,
            "rois": [
                {
                    "id": 3,
                    "rect": {"x1": 10, "y1": 20, "x2": 110, "y2": 120},
                }
            ],
        }

        response = client.post("/api/detection/regions", json=payload)
        assert response.status_code == 200

        roi = store.get_roi_config()["rois"][0]
        assert roi["target"] == "low"
        assert roi["type"] == "forbidden"
        assert roi["roi_type"] == "forbidden_zone"


def test_invalid_region_rejected():
    with TemporaryDirectory() as temp_dir:
        use_temp_configs(temp_dir)
        client = app.test_client()
        payload = {
            "target": "all",
            "clear": False,
            "regions": [
                {
                    "id": 1,
                    "rect": {"x1": 400, "y1": 200, "x2": 100, "y2": 350},
                }
            ],
        }

        response = client.post("/api/detection/regions", json=payload)
        assert response.status_code == 400
        assert response.get_json()["status"] == "error"


def test_project_roi_format_is_accepted():
    with TemporaryDirectory() as temp_dir:
        use_temp_configs(temp_dir)
        client = app.test_client()
        payload = {
            "target": "all",
            "clear": False,
            "regions": [
                {
                    "id": 2,
                    "roi_id": "warning_1",
                    "name": "预警区",
                    "enabled": True,
                    "roi_type": "warning_zone",
                    "judge_method": "foot_point",
                    "coordinate_mode": "normalized",
                    "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                    "overlap_thres": 0.2,
                }
            ],
        }

        response = client.post("/api/detection/regions", json=payload)
        assert response.status_code == 200

        roi = store.get_roi_config()["rois"][0]
        assert roi["roi_id"] == "warning_1"
        assert roi["type"] == "warning"
        assert roi["roi_type"] == "warning_zone"
        assert roi["coordinate_mode"] == "normalized"
        assert roi["judge_method"] == "foot_point"
        assert roi["polygon"] == [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


if __name__ == "__main__":
    test_detection_settings_aliases()
    test_detection_regions_rect_to_points_and_clear()
    test_detection_regions_accepts_rois_key()
    test_invalid_region_rejected()
    test_project_roi_format_is_accepted()
    print("✓ AI config API tests passed")
