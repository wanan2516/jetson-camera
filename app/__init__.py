from __future__ import annotations

from flask import Flask, jsonify, request

from .ai_config_store import (
    ConfigValidationError,
    ensure_config_files,
    get_detection_config,
    get_roi_config,
    save_detection_regions,
    save_detection_settings,
)


def create_app() -> Flask:
    flask_app = Flask(__name__)
    ensure_config_files()

    @flask_app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @flask_app.get("/api")
    def api_index():
        return jsonify({
            "status": "success",
            "message": "Camera AI config API is running",
        })

    @flask_app.post("/api/detection/settings")
    @flask_app.post("/api/detection/region")
    def update_detection_settings():
        data = request.get_json(silent=True)
        try:
            save_detection_settings(data)
            return jsonify({
                "status": "success",
                "message": "检测设置保存成功",
            })
        except ConfigValidationError as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 400

    @flask_app.post("/api/detection/regions")
    def update_detection_regions():
        data = request.get_json(silent=True)
        try:
            _, cleared = save_detection_regions(data)
            return jsonify({
                "status": "success",
                "message": "检测区域已清空" if cleared else "检测区域保存成功",
            })
        except ConfigValidationError as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 400

    @flask_app.get("/api/detection/config")
    def read_detection_config():
        try:
            return jsonify({
                "status": "success",
                "data": get_detection_config(),
            })
        except ConfigValidationError as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @flask_app.get("/api/detection/regions")
    def read_detection_regions():
        try:
            return jsonify({
                "status": "success",
                "data": get_roi_config(),
            })
        except ConfigValidationError as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    return flask_app


app = create_app()
