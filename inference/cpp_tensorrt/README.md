# Legacy C++ TensorRT Runtime

This directory preserves the original C++ TensorRT implementation from the
`jackxie0227/Cam_flaskvue` `jetson` branch.

The default runtime is now `inference/python_tensorrt`, which keeps the backend
`Model` interface but runs TensorRT through Python instead of a C++ subprocess.

