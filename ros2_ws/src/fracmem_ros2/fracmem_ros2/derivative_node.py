"""
Streams a scalar signal through a fitted fracmem
CompressedFractionalFilter online: subscribes std_msgs/Float64 on
`input_topic`, publishes the O(L+p)-per-sample derivative estimate as
std_msgs/Float64 on `output_topic`.

Fitting stays offline -- run this ahead of time and point the node at
the result:

    filt = CompressedFractionalFilter(alpha=0.5, h=0.01, L=32, p=16)
    filt.fit(train_signals)
    filt.save("filter.npz")

    ros2 run fracmem_ros2 derivative_node --ros-args -p filter_path:=/path/to/filter.npz
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from fracmem import CompressedFractionalFilter


class FractionalDerivativeNode(Node):
    def __init__(self):
        super().__init__("fracmem_derivative_node")
        self.declare_parameter("filter_path", "")
        self.declare_parameter("input_topic", "fracmem/input")
        self.declare_parameter("output_topic", "fracmem/output")

        filter_path = self.get_parameter("filter_path").get_parameter_value().string_value
        if not filter_path:
            raise RuntimeError(
                "Set the 'filter_path' parameter to a .npz saved by "
                "CompressedFractionalFilter.save(...) -- fit the filter offline first."
            )
        self.filt = CompressedFractionalFilter.load(filter_path)
        self.filt.reset_stream()
        self.get_logger().info(
            f"Loaded fracmem filter: alpha={self.filt.alpha}, h={self.filt.h}, "
            f"L={self.filt.L}, p={self.filt.p}, definition={self.filt.definition}"
        )

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._pub = self.create_publisher(Float64, output_topic, 10)
        self._sub = self.create_subscription(Float64, input_topic, self._on_input, 10)

    def _on_input(self, msg: Float64):
        y = self.filt.step(msg.data)
        out = Float64()
        out.data = float(y)
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FractionalDerivativeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
