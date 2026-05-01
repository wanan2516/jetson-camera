from pathlib import Path
import json
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


class Model:
    def __init__(
        self,
        config=None,
        executable_path=None,
        engine_path=None,
        work_dir=None,
        timeout=30,
    ):
        self.executable_path = self._resolve_executable(executable_path)
        self.engine_path = self._resolve_path(engine_path or "weights/yolo11n_person.engine")
        self.config_path = self._resolve_path(config or "configs/config.json")
        self.work_dir = self._resolve_path(work_dir or "tmp/inference")
        self.timeout = timeout
        self.last_result = None
        self.last_stdout = ""
        self.last_stderr = ""

        self._require_file(self.executable_path, "C++ TensorRT executable")
        self._require_file(self.engine_path, "TensorRT engine")
        self._require_file(self.config_path, "runtime config")

    def inference(self, frame, prestart=False):
        """
        输入：图像（np.ndarray）
        输出：C++ TensorRT 程序画好 ROI 和检测框后的图像（np.ndarray）
        """
        if frame is None:
            raise ValueError("frame must not be None")
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy.ndarray")
        if frame.size == 0:
            raise ValueError("frame must not be empty")
        if frame.ndim not in (2, 3):
            raise ValueError("frame must be a 2D grayscale or 3D color image")
        if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
            raise ValueError("frame must have 1, 3, or 4 channels")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.work_dir / "input.jpg"
        output_path = self.work_dir / "output.jpg"
        json_output_path = self.work_dir / "result.json"

        if not cv2.imwrite(str(input_path), frame):
            raise RuntimeError(f"failed to write input image: {input_path}")

        command = [
            str(self.executable_path),
            str(self.engine_path),
            str(input_path),
            str(output_path),
            str(self.config_path),
            str(json_output_path),
        ]
        if prestart:
            command.append("--prestart")

        self.last_result = None
        self.last_stdout = ""
        self.last_stderr = ""

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.CalledProcessError as exc:
            self.last_stdout = exc.stdout or ""
            self.last_stderr = exc.stderr or ""
            raise RuntimeError(
                "C++ TensorRT inference failed\n"
                f"stdout:\n{self.last_stdout}\n"
                f"stderr:\n{self.last_stderr}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            self.last_stdout = exc.stdout or ""
            self.last_stderr = exc.stderr or ""
            raise RuntimeError(
                f"C++ TensorRT inference timed out after {self.timeout} seconds\n"
                f"stdout:\n{self.last_stdout}\n"
                f"stderr:\n{self.last_stderr}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"failed to execute C++ TensorRT program: {exc}") from exc

        self.last_stdout = completed.stdout or ""
        self.last_stderr = completed.stderr or ""

        if json_output_path.exists():
            with json_output_path.open("r", encoding="utf-8") as result_file:
                self.last_result = json.load(result_file)

        result = cv2.imread(str(output_path))
        if result is None:
            raise RuntimeError(f"failed to read output image: {output_path}")
        return result

    @staticmethod
    def _resolve_path(path):
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        return resolved

    @classmethod
    def _resolve_executable(cls, executable_path):
        if executable_path is not None:
            return cls._resolve_path(executable_path)

        executable = ROOT / "cpp_tensorrt" / "build" / "camera_tensorrt"
        if executable.exists():
            return executable
        raise FileNotFoundError(f"C++ TensorRT executable not found: {executable}")

    @staticmethod
    def _require_file(path, label):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")


if __name__ == "__main__":
    image_path = ROOT / "assets" / "test.jpg"
    output_path = ROOT / "outputs" / "model_result.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"input image not found or unreadable: {image_path}")

    model = Model()
    result = model.inference(frame)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"failed to write output image: {output_path}")

    detections = model.last_result.get("detections", []) if model.last_result else []
    system_state = model.last_result.get("system_state") if model.last_result else None
    print(f"output: {output_path}")
    print(f"detections: {len(detections)}")
    print(f"system_state: {system_state}")
