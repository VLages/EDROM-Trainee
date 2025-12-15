import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from controller import Robot, Keyboard

# --- MENU DE SELEÇÃO INICIAL ---
print("=========================================")
print("      SELECIONE O MODO DE OPERAÇÃO       ")
print("=========================================")
print(" [1] Modo Follow ")
print(" [2] Modo Pick&Place ")
print(" [3] Modo Extra (Gravação de Poses)")
print("=========================================")

try:
    entrada_usuario = input("Digite a opção desejada (1, 2 ou 3): ").strip()
except Exception as e:
    print(f"Erro ao ler entrada: {e}")
    entrada_usuario = "0" 

if entrada_usuario == "2":
    print("\n>> Iniciando Modo Pick&Place <<\n")
    
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
            if self.robo.step(self.passo_tempo) == -1:
                return
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
                        cv2.drawContours(img_bgr, [maior_contorno], -1, (0, 255, 255), 2)
                        cv2.circle(img_bgr, (centro_x, centro_y), 5, (0, 255, 0), -1)
                        cv2.putText(img_bgr, f"X:{centro_x} Y:{centro_y}", (centro_x-20, centro_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                        mensagem = Float32MultiArray()
                        mensagem.data = [float(centro_x), float(centro_y), float(area_contorno)]
                        self.publicador.publish(mensagem)

            cv2.imshow("Camera UR5e", img_bgr)
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
                
                if loops_passados > 100: 
                    self.inicio_auto_loops = None
  
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

elif entrada_usuario == "3":
    print("\n>> Iniciando Modo Gravação <<\n")

    TIME_STEP = 32
    VEL_MANUAL = 1.0       # Velocidade ao mover com teclas
    VEL_PLAYBACK = 1.5     # Velocidade ao reproduzir o movimento
    TOLERANCIA = 0.05      # Precisão (radianos)
    DELAY_CLIQUE = 0.5     # [NOVO] Tempo de espera (segundos) para botões de comando

    class UR5eFinal:
        def __init__(self):
            self.robot = Robot()
            self.keyboard = Keyboard()
            self.keyboard.enable(TIME_STEP)

            # Configuração dos Motores
            self.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]
            
            self.motors = []
            self.sensors = []

            print("--> Configurando Hardware...")
            for name in self.joint_names:
                m = self.robot.getDevice(name)
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                self.motors.append(m)
                
                s = self.robot.getDevice(name + '_sensor')
                if s is None: s = m.getPositionSensor()
                if s:
                    s.enable(TIME_STEP)
                    self.sensors.append(s)

            # Variáveis de Estado
            self.poses = []
            self.estado = 'MANUAL'
            self.ultimo_tempo_cmd = 0.0  # Marca quando foi o último clique de comando

            print("=== SISTEMA PRONTO (COM DELAY) ===")
            print(" [WASD/QE] Mover Braço")
            print(" [P]       Mudar Modo (Manual <-> Gravação)")
            print(" [O]       Salvar Pose")
            print(" [F]       Finalizar e Tocar")

        def run(self):
            while self.robot.step(TIME_STEP) != -1:
                key = self.keyboard.getKey()
                agora = self.robot.getTime() # Tempo atual da simulação
                
                # --- COMANDO P: MUDANÇA DE MODO ---
                if key == ord('P'):
                    # Só aceita se passou o tempo do delay
                    if agora - self.ultimo_tempo_cmd > DELAY_CLIQUE:
                        self.alternar_modo()
                        self.ultimo_tempo_cmd = agora # Reseta o relógio do delay

                # --- LÓGICA DOS ESTADOS ---
                if self.estado == 'MANUAL':
                    self.control_manual(key)

                elif self.estado == 'GRAVANDO':
                    self.logic_gravacao(key, agora)

                elif self.estado == 'REPRODUZINDO':
                    pass 

        def alternar_modo(self):
            if self.estado == 'MANUAL':
                self.estado = 'GRAVANDO'
                self.poses = []
                print(f"\n>>> MODO GRAVAÇÃO <<<")
                # Para o robô
                for m in self.motors: m.setVelocity(0.0)
            
            elif self.estado == 'GRAVANDO' or self.estado == 'REPRODUZINDO':
                self.estado = 'MANUAL'
                print(f"\n<<< MODO MANUAL <<<")
                for m in self.motors: 
                    m.setPosition(float('inf'))
                    m.setVelocity(0.0)

        def logic_gravacao(self, key, agora):
            # Movimentação (Sem delay, fluida)
            self.control_manual(key)

            # COMANDO O: SALVAR POSE
            if key == ord('O'):
                if agora - self.ultimo_tempo_cmd > DELAY_CLIQUE:
                    pose_atual = [s.getValue() for s in self.sensors]
                    self.poses.append(pose_atual)
                    print(f" [+] Pose {len(self.poses)} Salva!")
                    self.ultimo_tempo_cmd = agora

            # COMANDO F: FINALIZAR
            elif key == ord('F'):
                if agora - self.ultimo_tempo_cmd > DELAY_CLIQUE:
                    self.ultimo_tempo_cmd = agora
                    
                    if len(self.poses) == 0:
                        print(" [!] Nada gravado para tocar.")
                    else:
                        self.estado = 'REPRODUZINDO'
                        print(f"\n>>> REPRODUZINDO {len(self.poses)} POSES... <<<")
                        self.executar_playback()
                        
                        # Ao terminar, volta para manual
                        self.estado = 'MANUAL'
                        print(">>> Fim. Voltando para Manual. <<<")
                        for m in self.motors:
                            m.setPosition(float('inf'))
                            m.setVelocity(0.0)

        def executar_playback(self):
            for i, pose in enumerate(self.poses):
                print(f" -> Pose {i+1}...", end=" ")
                
                # Envia motores
                for j, m in enumerate(self.motors):
                    m.setVelocity(VEL_PLAYBACK)
                    m.setPosition(pose[j])
                
                # Espera chegar
                chegou = False
                t_inicio = self.robot.getTime()
                
                while not chegou:
                    if self.robot.step(TIME_STEP) == -1: return

                    erro_max = 0
                    for j, s in enumerate(self.sensors):
                        diff = abs(s.getValue() - pose[j])
                        if diff > erro_max: erro_max = diff
                    
                    if erro_max < TOLERANCIA:
                        chegou = True
                        print("OK")
                    
                    # Timeout de segurança (4s)
                    if self.robot.getTime() - t_inicio > 4.0:
                        print("(Timeout)")
                        chegou = True

        def control_manual(self, key):
            vels = [0.0] * 6
            v = VEL_MANUAL
            
            if key == ord('A'): vels[0] = v
            elif key == ord('D'): vels[0] = -v
            elif key == ord('W'): vels[1] = v
            elif key == ord('S'): vels[1] = -v
            elif key == ord('Q'): vels[2] = v
            elif key == ord('E'): vels[2] = -v
            
            for i, m in enumerate(self.motors):
                m.setVelocity(vels[i])
                if self.estado != 'REPRODUZINDO':
                    m.setPosition(float('inf'))

    if __name__ == "__main__":
        app = UR5eFinal()
        app.run()

elif entrada_usuario == "1":
    print("\n>> Iniciando Modo Follow <<\n")

    class UR5VisualController(Node):
        def __init__(self):
            super().__init__('ur5_visual_controller')
            
            # --- 1. CONFIGURAÇÕES GERAIS ---
            self.MODEL_PATH = "best.pt"
            self.CONF_THRESHOLD = 0.50
            self.TARGET_CLASS_ID = 0  # 0 para modelo (bola de futebol tradicional) próprio, 32 para COCO (bola)
            self.YOLO_INTERVAL = 2    # Roda a IA a cada N frames
            
            # Ganhos do Controlador (P) / Velocidade dos motores
            self.KP = 1.2
            self.VEL_MANUAL = 1.0
            
            # Calibração de Direção dos Motores / Para ir na direção correta
            self.CALIB_BASE = 1.0
            self.CALIB_OMBRO = -1.0
            self.CALIB_COTOVELO = -1.0 
            
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
            
            print(" CONTROLADOR UR5 INICIADO")
            print(f"   Modo Inicial: MANUAL")
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
                print(f" Modo alterado para: {self.modos[self.modo_index]}")

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
            
            # Mapeamento de Teclas // Postivo = Esquerda/Cima , Negativo = Direita/Baixo
            if key == ord('A'): vels[0] = v     # Base 
            elif key == ord('D'): vels[0] = -v  # Base 
            elif key == ord('W'): vels[1] = v   # Ombro 
            elif key == ord('S'): vels[1] = -v  # Ombro 
            elif key == ord('Q'): vels[2] = v   # Cotovelo
            elif key == ord('E'): vels[2] = -v  # Cotovelo
            elif key == ord('T'): vels[3] = v   # Punho 1
            elif key == ord('G'): vels[3] = -v  # Punho 1
            elif key == ord('F'): vels[4] = v   # Punho 2
            elif key == ord('H'): vels[4] = -v  # Punho 2
            elif key == ord('R'): vels[5] = v   # Punho 3
            elif key == ord('Y'): vels[5] = -v  # Punho 3
            
            
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

else:
    print("Opção inválida ou não selecionada. O programa será encerrado.")