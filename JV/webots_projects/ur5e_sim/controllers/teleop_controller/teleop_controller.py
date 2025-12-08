from controller import Robot, Keyboard
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class UR5eNode(Node):
    def __init__(self, robot):
        super().__init__("ur5e_controller_node")

        self.robot = robot
        self.timestep = int(self.robot.getBasicTimeStep())

        
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        self.motors = [robot.getDevice(j) for j in self.joint_names]
        for m in self.motors:
            m.setVelocity(1.5)
            m.setPosition(float("inf"))  

        # limites reais do UR5e
        self.limits = [
            (-6.28, 6.28),
            (-2.09, 2.09),
            (-3.14, 3.14),
            (-6.28, 6.28),
            (-6.28, 6.28),
            (-6.28, 6.28),
        ]

        self.joint_positions = [0.0] * 6
        self.delta = 0.03         
        self.current_joint = 0

        # ----------- KEYBOARD -----------
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # ----------- CAMERA -----------
        self.camera = robot.getDevice("camera")
        self.camera.enable(self.timestep)

        # ----------- ROS2 -----------
        self.publisher = self.create_publisher(
            Float32MultiArray,
            "object_coords",
            10
        )

        self.get_logger().info("UR5e Controller iniciado com sucesso!")
        self.get_logger().info("Teleop ativo (1..6 para juntas, +/- para mover)")

    # ============= TELEOP ==================
    def handle_keyboard(self):
        key = self.keyboard.getKey()
        if key == -1:
            return

        # selecionar junta 1..6
        if key in [49, 50, 51, 52, 53, 54]:
            self.current_joint = key - 49
            self.get_logger().info(
                f"Junta {self.current_joint + 1} selecionada ({self.joint_names[self.current_joint]})"
            )
            return

        # incremento '+' (43), decremento '-' (45)
        if key in [43, 45]:
            low, high = self.limits[self.current_joint]

            x = self.joint_positions[self.current_joint]

            if key == 43:   # '+'
                x += self.delta
            else:           # '-'
                x -= self.delta

            # clamping profissional
            x = max(low, min(high, x))
            self.joint_positions[self.current_joint] = x

        # espaço para reset
        elif key == 32:
            self.joint_positions = [0.0] * 6
            self.get_logger().info("Todas as juntas resetadas.")

    # aplicar posições (modo velocidade)
    def apply_positions(self):
        for i, motor in enumerate(self.motors):
            motor.setPosition(self.joint_positions[i])

    # ============= VISÃO ==================
    def process_camera(self):
        img = self.camera.getImageArray()
        if img is None:
            return

        frame = np.array(img, dtype=np.uint8)
        frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)

        # publicar no ROS2
        msg = Float32MultiArray()
        msg.data = [float(x), float(y)]
        self.publisher.publish(msg)

    # ============= LOOP ====================
    def run(self):
        while self.robot.step(self.timestep) != -1:
            self.handle_keyboard()
            self.apply_positions()
            self.process_camera()


def main():
    rclpy.init()
    robot = Robot()
    node = UR5eNode(robot)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()