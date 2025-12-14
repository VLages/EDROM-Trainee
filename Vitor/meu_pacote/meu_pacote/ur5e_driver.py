import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from controller import Robot, Keyboard


class ControladorUR5e(Node):
    def __init__(self, robo_webots):
        super().__init__("nodo_controlador_ur5e")
        self.robo = robo_webots

        # Variáves do modo automático
        self.modo_automatico = False
        self.apertar_espaço = False
        
        # Configuração do Webots
        self.passo_tempo = int(self.robo.getBasicTimeStep())
        self.distancia_sensor = self.robo.getDevice("distance sensor")
        self.distancia_sensor.enable(self.passo_tempo)
        
        # Variáveis de controle de tempo
        self.contador_loops = 0         
        self.inicio_auto_loops = None  
        self.max_patience = 20 
        self.frame_counter = 0
        self.target_cx = None
        self.target_cy = None
        self.patience = 0
        self.modo_index = 0
        
        # Configuração do Timer do ROS 2
        periodo_em_segundos = self.passo_tempo / 1000.0
        self.create_timer(periodo_em_segundos, self.executar_modo_automatico)

        # Motores e seus nomes / movimentos
        self.nomes_motores = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.nomes_motores_garra = [
            "finger_1_joint_1",
            "finger_2_joint_1",
            "finger_middle_joint_1"
        ]
        self.motores_total = self.nomes_motores + self.nomes_motores_garra
        self.lista_motores = []
        for nome in self.motores_total:
            motor = self.robo.getDevice(nome)
            if motor:
                motor.setVelocity(1.5)
                self.lista_motores.append(motor)
            else:
                self.get_logger().error(f"Motor {nome} não encontrado!")    
        self.limites = [
            (-6.28, 6.28),
            (-2.09, 2.09),
            (-3.14, 3.14),
            (-6.28, 6.28),
            (-6.28, 6.28),
            (-6.28, 6.28),
        ]
        self.limites_garra = [
            (0.05, 1),
            (0.05, 1),
            (0.05, 1)
        ]
        self.posicoes_alvo = [0.0] * 9
        self.posicoes_alvo_garra = [0.05] *9
        self.passo_movimento = 0.05      
        self.motor_selecionado = 0  

        # Teclado
        self.teclado = Keyboard()
        self.teclado.enable(self.passo_tempo)

        # Camera YOLO
        self.MODEL_PATH = "best.pt"
        self.CONF_THRESHOLD = 0.50
        self.TARGET_CLASS_ID = 0 
        self.YOLO_INTERVAL = 2
        self.camera = self.robo.getDevice('camera')
        if not self.camera:
            self.camera = self.robo.getDevice('camera_sensor')
        if self.camera:
            self.camera.enable(self.passo_tempo)
        print(f" Carregando modelo: {self.MODEL_PATH}") 
        self.model = YOLO(self.MODEL_PATH)

        # Publicador ROS2
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_coords = self.create_publisher(Point, '/vision/object_coords', 10)
        self.bridge = CvBridge()

        self.get_logger().info("Controlador UR5e iniciado com sucesso!")

    # TECLADO
    def ler_teclado(self):
        tecla = self.teclado.getKey()
        if tecla == -1: return
    
        # Ativar modo Pick-Place automatico (Space bar)
        if tecla in [32]:
            self.modo_automatico = True
            if self.modo_automatico:
                self.get_logger().info("Modo Automático: ATIVADO")
                self.posicoes_alvo = [0.0] * 6
                self.posicoes_alvo_garra = [0.05] * 6
            return
        
        # Desativar modo Pick-Place automatico (M)
        if tecla in [77]:
            self.modo_automatico = False
            self.get_logger().info("Modo Automático: DESATIVADO")
            return

        # Selecionar qual motor controlar (Teclas 1 a 9)
        if tecla in [49, 50, 51, 52, 53, 54, 55, 56, 57]:
            self.motor_selecionado = tecla - 49
            self.get_logger().info(f"Motor {self.motor_selecionado + 1} selecionado")
            return

        # Movimentar (+ ou -)
        if tecla in [ord('+'), 43, 61, ord('-'), 45, 95]: 
            
            if self.motor_selecionado >= 6:
                indice_na_garra = self.motor_selecionado - 6
                
                minimo, maximo = self.limites_garra[indice_na_garra]
                posicao_atual = self.posicoes_alvo_garra[indice_na_garra]
                
                if tecla in [ord('+'), 43, 61]:
                    posicao_atual += self.passo_movimento
                else: 
                    posicao_atual -= self.passo_movimento
                
                posicao_atual = max(minimo, min(maximo, posicao_atual))
                self.posicoes_alvo_garra[indice_na_garra] = posicao_atual 

            else:
                minimo, maximo = self.limites[self.motor_selecionado]
                posicao_atual = self.posicoes_alvo[self.motor_selecionado]
                
                if tecla in [ord('+'), 43, 61]: 
                    posicao_atual += self.passo_movimento
                else: 
                    posicao_atual -= self.passo_movimento
                    
                posicao_atual = max(minimo, min(maximo, posicao_atual))
                self.posicoes_alvo[self.motor_selecionado] = posicao_atual
        
        elif tecla == 48:
            self.posicoes_alvo = [0.0] * 6         
            self.posicoes_alvo_garra = [0.05] * 6  
            self.get_logger().info("Posições resetadas.")

    # MOVER O ROBÔ 
    def aplicar_movimento(self):
        for i, motor in enumerate(self.lista_motores):
            if i < 6:
                posicao = self.posicoes_alvo[i]
                motor.setPosition(posicao)
            else:
                indice_garra = i - 6
                if indice_garra < len(self.posicoes_alvo_garra):
                    posicao = self.posicoes_alvo_garra[indice_garra]
                    motor.setPosition(posicao)
    
    # PROCESSAMENTO DE VISÂO
    def processar_camera(self):
        if self.robo.step(self.passo_tempo) == -1:
            return
        img_w, img_h = 0, 0
        
        if self.camera:
            raw_img = self.camera.getImage()
            if raw_img:
                img_w = self.camera.getWidth()
                img_h = self.camera.getHeight()
                
                # Conversão Webots -> OpenCV
                np_img = np.frombuffer(raw_img, np.uint8).reshape((img_h, img_w, 4))
                img_bgr = np_img[:, :, :3].copy()
                
                self.contador_loops += 1
                
                # Inferência (Frame Skipping para performance)
                if self.contador_loops % self.YOLO_INTERVAL == 0:
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
                    color = (0, 0, 255) if self.contador_loops % self.YOLO_INTERVAL == 0 else (0, 255, 255)
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

    # EXECUTAR MODO AUTOMATICO
    def executar_modo_automatico(self):
        if self.robo.step(self.passo_tempo) == -1:
            return
        self.contador_loops += 1

        if not self.modo_automatico:
            return
        ocupado = self.inicio_auto_loops is not None

        if not ocupado:
            valor_sensor = self.distancia_sensor.getValue()

            if valor_sensor < 250:
                self.inicio_auto_loops = self.contador_loops
                self.posicoes_alvo_garra = [0.5] * 6
            
            else:
                self.posicoes_alvo_garra = [0.05] * 6
                
        else:
            loops_passados = self.contador_loops - self.inicio_auto_loops
            if loops_passados < 40:
                self.posicoes_alvo_garra = [0.5] * 6

            if loops_passados > 10: 
                self.posicoes_alvo[1] = -2

            if loops_passados > 15:
                self.posicoes_alvo[2] = -2

            if loops_passados > 40:
                self.posicoes_alvo_garra = [0.05] * 6
            
            if loops_passados > 60:
                self.posicoes_alvo = [0.0] * 6
                
    def executar(self):
        while self.robo.step(self.passo_tempo) != -1 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            self.ler_teclado()
            self.executar_modo_automatico()
            self.aplicar_movimento()
            self.processar_camera()

def main(args=None):
    rclpy.init(args=args)
    meu_robo = Robot()
    nodo = ControladorUR5e(meu_robo)
    
    nodo.executar()
    
    nodo.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
    
main()

    