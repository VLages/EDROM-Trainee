import rclpy
from rclpy.node import Node
from controller import Robot, Keyboard 
import cv2
import numpy as np
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image 
from geometry_msgs.msg import Point 

class UR5ManualFollow(Node):
    def __init__(self):
        super().__init__('ur5_clean_node')
        
        # --- 1. INICIALIZAÇÃO DO ROBÔ ---
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        
        # ### CRIAÇÃO DOS PUBLICADORES ROS2 ###
        # Cria o tópico onde a imagem da câmera será enviada
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        # Cria o tópico para cumprir a Meta 3.c (Coordenadas da bola)
        self.pub_coords = self.create_publisher(Point, '/vision/object_coords', 10)
        # Ferramenta de conversão OpenCV -> ROS
        self.bridge = CvBridge()
        
        # --- 2. CONFIGURAÇÃO DOS MOTORES ---
        self.motor_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.motors = []
        for name in self.motor_names:
            m = self.robot.getDevice(name)
            if m:
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                self.motors.append(m)
            else:
                self.get_logger().error(f"Motor {name} não encontrado!")

        # --- 3. CONFIGURAÇÃO DA CÂMERA ---
        self.camera = self.robot.getDevice('camera')
        if not self.camera: self.camera = self.robot.getDevice('camera_sensor')
        if self.camera: self.camera.enable(self.timestep)

        # --- 4. VARIÁVEIS DE CONTROLE ---
        self.modos = ['MANUAL', 'FOLLOW']
        self.modo_index = 0
        self.kp = 2.0
        self.vel_manual = 1.0
        self.last_toggle_time = 0 
        self.calib_x = 1.0
        self.calib_y = -1.0

        self.create_timer(self.timestep / 1000.0, self.run_loop)
        
        print("CONTROLADOR UR5e MANUAL/FOLLOW INICIADO")
        print("[M] Alternar Modo | [WASD/QE-TFGH/RY] Controle Manual")

    def run_loop(self):
        if self.robot.step(self.timestep) == -1: rclpy.shutdown(); return

        # ==========================================================
        # PARTE 1: PROCESSAMENTO DE VISÃO E PUBLICAÇÃO ROS
        # ==========================================================
        cx, cy, img_w, img_h = None, None, 0, 0
        
        if self.camera:
            raw = self.camera.getImage()
            if raw:
                img_w = self.camera.getWidth()
                img_h = self.camera.getHeight()
                img = np.frombuffer(raw, np.uint8).reshape((img_h, img_w, 4))
                img_bgr = img[:, :, :3].copy()
                
                ### PUBLICAR A IMAGEM NO TÓPICO ROS ###
                # Converte a imagem BGR do OpenCV para mensagem ROS e publica
                msg_image = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
                self.pub_image.publish(msg_image)

                # Processamento da imagem (Manteve igual)
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([0, 50, 30]), np.array([15, 255, 255])) + \
                       cv2.inRange(hsv, np.array([160, 50, 30]), np.array([180, 255, 255]))
                
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=2)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 100:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(img_bgr, (cx, cy), 5, (0, 255, 0), -1)
                            
                            ### PUBLICAR COORDENADAS (META 3.c) ###
                            point_msg = Point()
                            point_msg.x = float(cx)
                            point_msg.y = float(cy)
                            point_msg.z = 0.0
                            self.pub_coords.publish(point_msg)

                cv2.putText(img_bgr, f"MODO: {self.modos[self.modo_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Visao do Robo", img_bgr)
                cv2.waitKey(1)

        # ==========================================================
        # Controles Manual e Follow
        # ==========================================================
        key = self.keyboard.getKey()
        now = self.get_clock().now().nanoseconds / 1e9
        
        if key == ord('M') and (now - self.last_toggle_time > 0.5):
            self.last_toggle_time = now
            for m in self.motors: m.setVelocity(0.0)
            self.modo_index = 1 if self.modo_index == 0 else 0
            print(f"🔄 MUDANDO PARA: {self.modos[self.modo_index]}")

        modo_atual = self.modos[self.modo_index]

        if modo_atual == 'MANUAL':
            val = self.vel_manual
            vels = [0.0] * 6 
            if key == ord('A'): vels[0] = val
            elif key == ord('D'): vels[0] = -val
            elif key == ord('W'): vels[1] = val
            elif key == ord('S'): vels[1] = -val
            elif key == ord('Q'): vels[2] = val
            elif key == ord('E'): vels[2] = -val
            elif key == ord('T'): vels[3] = val
            elif key == ord('G'): vels[3] = -val
            elif key == ord('F'): vels[4] = val
            elif key == ord('H'): vels[4] = -val
            elif key == ord('R'): vels[5] = val
            elif key == ord('Y'): vels[5] = -val
            
            for i, m in enumerate(self.motors):
                m.setPosition(float('inf'))
                m.setVelocity(vels[i])

        elif modo_atual == 'FOLLOW':
            if cx is not None:
                erro_x = ( (img_w / 2) - cx ) / img_w
                erro_y = ( (img_h / 2) - cy ) / img_h
                motor_base = erro_x * self.kp * self.calib_x
                motor_ombro = erro_y * self.kp * self.calib_y
                self.motors[0].setPosition(float('inf'))
                self.motors[0].setVelocity(motor_base)
                self.motors[1].setPosition(float('inf'))
                self.motors[1].setVelocity(motor_ombro)
            else:
                self.motors[0].setVelocity(0.0)
                self.motors[1].setVelocity(0.0)

def main(args=None):
    rclpy.init(args=args)
    node = UR5ManualFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
