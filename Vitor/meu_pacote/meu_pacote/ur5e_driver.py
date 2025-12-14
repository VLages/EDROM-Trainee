import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from controller import Robot, Keyboard, DistanceSensor

class ControladorUR5e(Node):
    def __init__(self, robo_webots):
        super().__init__("nodo_controlador_ur5e")
        self.modo_automatico = False
        self.apertar_espaço = False
        self.robo = robo_webots
        
        # Configuração do Webots
        self.passo_tempo = int(self.robo.getBasicTimeStep())
        self.distancia_sensor = self.robo.getDevice("distance sensor")
        self.distancia_sensor.enable(self.passo_tempo)
        
        # Variáveis de controle de tempo (Contador de Loops)
        self.contador_loops = 0         
        self.inicio_auto_loops = None   
        
        # Configuração do Timer do ROS 2
        periodo_em_segundos = self.passo_tempo / 1000.0
        self.create_timer(periodo_em_segundos, self.executar_modo_automatico)

        
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

        # TECLADO
        self.teclado = Keyboard()
        self.teclado.enable(self.passo_tempo)

        # CÂMERA
        self.camera = self.robo.getDevice("camera")
        if not self.camera:
            self.camera = self.robo.getDevice("camera_fixa")
            
        if self.camera:
            self.camera.enable(self.passo_tempo)
            self.get_logger().info(f"Câmera iniciada: {self.camera.getName()}")
        else:
            self.get_logger().warn("NENHUMA CÂMERA ENCONTRADA!")

        # ROS2 Publicador
        self.publicador = self.create_publisher(
            Float32MultiArray,
            "coordenadas_objeto", 
            10
        )

        self.get_logger().info("Controlador UR5e iniciado com sucesso!")

    # TECLADO
    def ler_teclado(self):
        tecla = self.teclado.getKey()
        if tecla == -1: return
    
    # Ativar modo Pick-Place automatico
        if tecla in [32]:
            self.modo_automatico = True
            if self.modo_automatico:
                self.get_logger().info("Modo Automático: ATIVADO")
                self.posicoes_alvo = [0.0] * 6
                self.posicoes_alvo_garra = [0.05] * 6
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
    
    # PROCESSAMENTO DE IMAGEM 
    def processar_camera(self):
        if not self.camera: return

        imagem_bruta = self.camera.getImage()
        if not imagem_bruta: return

        largura = self.camera.getWidth()
        altura = self.camera.getHeight()
        
        img_matriz = np.frombuffer(imagem_bruta, np.uint8).reshape((altura, largura, 4))
        img_bgr = img_matriz[:, :, :3].copy() 

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Faixas de cor vermelha
        vermelho_min1 = np.array([0, 100, 100])
        vermelho_max1 = np.array([10, 255, 255])
        vermelho_min2 = np.array([170, 100, 100])
        vermelho_max2 = np.array([180, 255, 255])
        
        # Criar máscara (Preto e Branco)
        mascara = cv2.inRange(img_hsv, vermelho_min1, vermelho_max1) + cv2.inRange(img_hsv, vermelho_min2, vermelho_max2)

        # Achar contornos
        contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            # Pegar o maior contorno (para ignorar ruídos)
            maior_contorno = max(contornos, key=cv2.contourArea)
            area_contorno = cv2.contourArea(maior_contorno)

            if area_contorno > 100:
                momentos = cv2.moments(maior_contorno)
                if momentos["m00"] != 0:
                    # Calcular centro X e Y
                    centro_x = int(momentos["m10"] / momentos["m00"])
                    centro_y = int(momentos["m01"] / momentos["m00"])

                    # Desenhar na tela
                    cv2.drawContours(img_bgr, [maior_contorno], -1, (255, 255, 255), 2)
                    cv2.circle(img_bgr, (centro_x, centro_y), 5, (0, 0, 0), -1)
                    cv2.putText(img_bgr, f"X:{centro_x} Y:{centro_y}", (centro_x-20, centro_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    mensagem = Float32MultiArray()
                    mensagem.data = [float(centro_x), float(centro_y), float(area_contorno)]
                    self.publicador.publish(mensagem)

        cv2.imshow("Camera UR5e", img_bgr)
        cv2.waitKey(1)

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
                self.posicoes_alvo[2] = -1.5

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