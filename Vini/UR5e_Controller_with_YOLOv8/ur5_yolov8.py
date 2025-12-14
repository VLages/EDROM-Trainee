import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# Importações do Webots
from controller import Robot, Keyboard

class UR5VisualController(Node):
    def __init__(self):
        super().__init__('ur5_visual_controller')
        
        # --- 1. CONFIGURAÇÕES GERAIS ---
        self.MODEL_PATH = "best.pt"
        self.CONF_THRESHOLD = 0.50
        self.TARGET_CLASS_ID = 0  # 0 para modelo custom, 32 para COCO (bola)
        self.YOLO_INTERVAL = 2    # Roda a IA a cada N frames
        
        # Ganhos do Controlador (P)
        self.KP = 1.2
        self.VEL_MANUAL = 1.0
        
        # Calibração de Direção dos Motores
        self.CALIB_BASE = 1.0
        self.CALIB_OMBRO = -1.0
        self.CALIB_COTOVELO = -1.0  # Configurado para alcance longo
        
        # --- 2. INICIALIZAÇÃO DO ROBÔ (WEBOTS) ---
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # Inicialização dos Motores
        self.motor_names = [
            'shoulder_pan_joint',  # [0] Base
            'shoulder_lift_joint', # [1] Ombro
            'elbow_joint',         # [2] Cotovelo
            'wrist_1_joint',       # [3] Punho 1
            'wrist_2_joint',       # [4] Punho 2
            'wrist_3_joint'        # [5] Punho 3
        ]
        self.motors = []
        for name in self.motor_names:
            motor = self.robot.getDevice(name)
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
            self.motors.append(motor)


        # Inicialização da Câmera
        self.camera = self.robot.getDevice('camera')
        if not self.camera:
            self.camera = self.robot.getDevice('camera_sensor')
        if self.camera:
            self.camera.enable(self.timestep)

        # --- 3. INICIALIZAÇÃO DA IA (YOLO) ---
        print(f" Carregando modelo: {self.MODEL_PATH}") 
        self.model = YOLO(self.MODEL_PATH)
        

        # Variáveis de Controle de Fluxo
        self.frame_counter = 0
        self.target_cx = None
        self.target_cy = None
        self.patience = 0
        self.max_patience = 20
        self.modo_index = 0 # 0 = Manual, 1 = Follow
        self.modos = ['MANUAL', 'FOLLOW']
        self.last_toggle_time = 0

        # --- 4. ROS2 PUBLISHERS ---
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_coords = self.create_publisher(Point, '/vision/object_coords', 10)
        self.bridge = CvBridge()

        # Timer Principal
        self.create_timer(self.timestep / 1000.0, self.run_loop)
        
        print("✅ CONTROLADOR UR5 INICIADO")
        print(f"   Modo Inicial: {self.modos[self.modo_index]}")
        print("   Comandos: [M] Alternar Modo | [WASD/QE] Controle Manual")

    def run_loop(self):
        # Passo de simulação do Webots
        if self.robot.step(self.timestep) == -1:
            rclpy.shutdown()
            return

        # ==========================================================
        # 1. PROCESSAMENTO DE VISÃO (YOLO)
        # ==========================================================
        img_w, img_h = 0, 0
        
        if self.camera:
            raw_img = self.camera.getImage()
            if raw_img:
                img_w = self.camera.getWidth()
                img_h = self.camera.getHeight()
                
                # Conversão Webots -> OpenCV
                np_img = np.frombuffer(raw_img, np.uint8).reshape((img_h, img_w, 4))
                img_bgr = np_img[:, :, :3].copy()
                
                self.frame_counter += 1
                
                # Inferência (Frame Skipping para performance)
                if self.frame_counter % self.YOLO_INTERVAL == 0:
                    results = self.model(img_bgr, verbose=False)
                    found_now = False
                    
                    for result in results:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if conf > self.CONF_THRESHOLD and cls_id == self.TARGET_CLASS_ID:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                self.target_cx = (x1 + x2) // 2
                                self.target_cy = (y1 + y2) // 2
                                
                                self.patience = self.max_patience
                                found_now = True
                                
                                # Visualização
                                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
                                cv2.putText(img_bgr, f"Bola {conf:.2f}", (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                                break
                        if found_now: break
                    
                    if not found_now:
                        self.patience -= 1

                # Lógica de Persistência (Memória)
                if self.patience > 0 and self.target_cx is not None:
                    # Desenha o alvo (Vermelho = Atual, Amarelo = Memória)
                    color = (0, 0, 255) if self.frame_counter % self.YOLO_INTERVAL == 0 else (0, 255, 255)
                    cv2.circle(img_bgr, (self.target_cx, self.target_cy), 5, color, -1)
                    
                    # Publica Coordenadas no ROS
                    point_msg = Point()
                    point_msg.x = float(self.target_cx)
                    point_msg.y = float(self.target_cy)
                    self.pub_coords.publish(point_msg)
                else:
                    self.target_cx = None

                # Publica Imagem e Mostra Janela
                msg_image = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
                self.pub_image.publish(msg_image)
                cv2.imshow("UR5 Vision", img_bgr)
                cv2.waitKey(1)

        # ==========================================================
        # 2. LÓGICA DE CONTROLE
        # ==========================================================
        key = self.keyboard.getKey()
        now = self.get_clock().now().nanoseconds / 1e9
        
        # Alternância de Modo
        if key == ord('M') and (now - self.last_toggle_time > 0.5):
            self.last_toggle_time = now
            # Para os motores antes de trocar
            for m in self.motors: m.setVelocity(0.0)
            self.modo_index = 1 if self.modo_index == 0 else 0
            print(f"🔄 Modo alterado para: {self.modos[self.modo_index]}")

        modo_atual = self.modos[self.modo_index]

        if modo_atual == 'MANUAL':
            self._controle_manual(key)
        elif modo_atual == 'FOLLOW':
            self._controle_follow(img_w, img_h)

    # --- Funções Auxiliares de Controle ---

    def _controle_manual(self, key):
        """Gerencia a movimentação pelas teclas WASD/QE/RF..."""
        vels = [0.0] * 6
        v = self.VEL_MANUAL
        
        # Mapeamento de Teclas
        if key == ord('A'): vels[0] = v   # Base Esquerda
        elif key == ord('D'): vels[0] = -v  # Base Direita
        elif key == ord('W'): vels[1] = v   # Ombro Baixo
        elif key == ord('S'): vels[1] = -v  # Ombro Cima
        elif key == ord('Q'): vels[2] = v   # Cotovelo
        elif key == ord('E'): vels[2] = -v
        # Adicione mais teclas se necessário para os punhos
        
        for i, m in enumerate(self.motors):
            if i < len(vels):
                m.setVelocity(vels[i])

    def _controle_follow(self, img_w, img_h):
        """Controlador Proporcional para seguir o objeto"""
        if self.target_cx is not None and img_w > 0:
            # Cálculo do Erro Normalizado (-0.5 a +0.5)
            erro_x = ((img_w / 2) - self.target_cx) / img_w
            erro_y = ((img_h / 2) - self.target_cy) / img_h
            
            # Aplicação dos Ganhos e Calibração
            vel_base = erro_x * self.KP * self.CALIB_BASE
            vel_ombro = erro_y * self.KP * self.CALIB_OMBRO
            vel_cotovelo = erro_y * self.KP * self.CALIB_COTOVELO
            
            # Aplica velocidades
            self.motors[0].setVelocity(vel_base)
            self.motors[1].setVelocity(vel_ombro)
            self.motors[2].setVelocity(vel_cotovelo)
            

        else:
            # Se perder o alvo, para tudo
            for m in self.motors:
                m.setVelocity(0.0)

def main(args=None):
    rclpy.init(args=args)
    node = UR5VisualController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
