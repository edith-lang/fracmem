import pytest

rclpy = pytest.importorskip("rclpy")

from fracmem_ros2.derivative_node import FractionalDerivativeNode  # noqa: E402


def test_missing_filter_path_raises():
    rclpy.init()
    try:
        with pytest.raises(RuntimeError):
            FractionalDerivativeNode()
    finally:
        rclpy.shutdown()
