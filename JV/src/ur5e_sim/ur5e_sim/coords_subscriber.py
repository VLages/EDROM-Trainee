import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class CoordsSubscriber(Node):
    def __init__(self):
        super().__init__('coords_subscriber')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            'object_coords',
            self.callback,
            10
        )

        self.get_logger().info("Subscriber /object_coords iniciado.")

    def callback(self, msg):
        x, y = msg.data
        self.get_logger().info(f"Alvo detectado em x={x:.2f}, y={y:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = CoordsSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()