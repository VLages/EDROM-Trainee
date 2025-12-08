import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from controller import Robot, Keyboard

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

        self.motors = []
        for name in self.joint_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setVelocity(1.5)
                self.motors.append(motor)
            else:
                self.get_logger().error(f"Motor {name} não encontrado!")

        self.limits = [
            (-6.28, 6.28),
            (-2.09, 2.09),
            (-3.14, 3.14),
            (-6.28, 6.28),
            (-6.28, 6.28),
            (-6.28, 6.28)
        ]

        self.joint_positions = [0.0] * 6
        self.delta = 0.05          
        self.current_joint = 0

        # KEYBOARD
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # CAMERA DETEC
        self.camera = self.robot.getDevice("camera")
        if not self.camera:
            self.camera = self.robot.getDevice("camera_fixa")
            
        if self.camera:
            self.camera.enable(self.timestep)
            self.get_logger().info(f"Câmera iniciada: {self.camera.getName()}")
        else:
            self.get_logger().warn("NENHUMA CÂMERA ENCONTRADA!")

        # ROS2
        self.publisher = self.create_publisher(
            Float32MultiArray,
            "object_coords",
            10
        )

        self.get_logger().info("UR5e Controller iniciado com sucesso!")

    # TELEOP
    def handle_keyboard(self):
        key = self.keyboard.getKey()
        if key == -1: return

        # Selecionar junta 1..6
        if key in [49, 50, 51, 52, 53, 54]:
            self.current_joint = key - 49
            self.get_logger().info(f"Junta {self.current_joint + 1} selecionada")
            return

        # Movimento (+/-)
        if key in [ord('+'), 43, 61, ord('-'), 45, 95]: 
            low, high = self.limits[self.current_joint]
            x = self.joint_positions[self.current_joint]

            if key in [ord('+'), 43, 61]: x += self.delta
            else: x -= self.delta

            # Clamping
            x = max(low, min(high, x))
            self.joint_positions[self.current_joint] = x

        # Espaço para reset
        elif key == 32:
            self.joint_positions = [0.0] * 6
            self.get_logger().info("Posições resetadas.")

    # MOTORES
    def apply_positions(self):
        for i, motor in enumerate(self.motors):
            motor.setPosition(self.joint_positions[i])

    # VISÃO
    def process_camera(self):
        if not self.camera: return

        # Captura 
        raw_image = self.camera.getImage()
        if not raw_image: return

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        
        img = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        
        img_bgr = img[:, :, :3].copy()

        # (Vermelho)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 100, 100])
        upper2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

        # Contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 100:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Desenho
                    cv2.drawContours(img_bgr, [c], -1, (0, 255, 255), 2)
                    cv2.circle(img_bgr, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(img_bgr, f"X:{cx} Y:{cy}", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                    # Publicar ROS2
                    msg = Float32MultiArray()
                    msg.data = [float(cx), float(cy), float(area)]
                    self.publisher.publish(msg)

        cv2.imshow("UR5e Camera", img_bgr)
        cv2.waitKey(1)

    def run(self):
        while self.robot.step(self.timestep) != -1 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            self.handle_keyboard()
            self.apply_positions()
            self.process_camera()

def main(args=None):
    rclpy.init(args=args)
    robot = Robot()
    node = UR5eNode(robot)
    
    node.run()
    
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

main()
